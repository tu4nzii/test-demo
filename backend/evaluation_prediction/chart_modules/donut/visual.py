"""Visual overlays and crop helpers for donut-chart prediction flows."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from PIL import Image, ImageDraw, ImageFont

from ...common.amplifier_style import amplifier_max_side, amplifier_target_side
from .model import call_llm_once

AMPLIFIER_OUTPUT_ROOT: str | None = None


def safe_print(*values) -> None:
    text = " ".join(str(value) for value in values)
    try:
        sys.stdout.write(text + "\n")
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def draw_angle_feedback(
    image_path: str,
    angle_deg: float | list[float],
    output_path: str,
    circle_center: tuple[int, int],
    inner_radius: int,
    grid_line_ratio: float = 0.1,
    line_color: tuple = (255, 0, 0, 255),
    line_width: int = 3,
    connect_arc: bool = True
) -> str:
    """
    在 donut 图上绘制反馈标记（新版角度系）：
    - 输入角度：顺时针，0° 在正上方（12 点方向）
      0°=上, 90°=右, 180°=下, 270°=左
    - 每个角度：画径向线 + 小弧
    - 两个角度：额外画完整弧线（按顺时针从 start → end）
    """

    import os, math
    from PIL import Image, ImageDraw

    # --- 输出目录 ---
    base_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    filename = _sanitize_filename(filename)
    feedback_img_dir = os.path.join(base_dir, "feedback_img")
    os.makedirs(feedback_img_dir, exist_ok=True)
    output_path = os.path.join(feedback_img_dir, filename)

    # --- 标准化角度输入 ---
    if isinstance(angle_deg, (int, float)):
        angle_list = [float(angle_deg)]
    elif isinstance(angle_deg, list) and all(isinstance(a, (int, float)) for a in angle_deg):
        angle_list = [float(a) for a in angle_deg]
    else:
        raise ValueError(f"Invalid angle_deg: {angle_deg}")

    # --- 打开图像 ---
    with Image.open(image_path).convert("RGBA") as base:
        img = base.copy()
    draw = ImageDraw.Draw(img)

    cx, cy = circle_center
    # 中心点
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 255, 0, 255))

    grid_line_length = int(inner_radius * grid_line_ratio)

    # ⭐ 统一角度变换：
    #   输入 angle_new：0°在上，顺时针
    #   对应绘制用角度 θ：从右侧顺时针，满足：
    #   θ = angle_new - 90（度）
    def to_draw_angle(angle_new: float) -> float:
        return (angle_new - 90.0) % 360.0

    # --- 每个角度：径向线 + 小弧 ---
    for angle in angle_list:
        draw_angle_deg = to_draw_angle(angle)          # 用于 cos/sin & arc
        theta = math.radians(draw_angle_deg)

        # 径向线
        x_in = cx + (inner_radius - grid_line_length) * math.cos(theta)
        y_in = cy + (inner_radius - grid_line_length) * math.sin(theta)
        x_out = cx + (inner_radius + grid_line_length) * math.cos(theta)
        y_out = cy + (inner_radius + grid_line_length) * math.sin(theta)
        draw.line([(x_in, y_in), (x_out, y_out)], fill=line_color, width=line_width)

        # 小弧（以 inner_radius 为半径，在 angle 附近画一小段）
        r_mid = inner_radius
        delta_theta = (grid_line_length / r_mid)
        delta_deg = math.degrees(delta_theta)

        start_deg = (draw_angle_deg - delta_deg) % 360
        end_deg   = (draw_angle_deg + delta_deg) % 360
        bbox = [cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid]
        draw.arc(bbox, start=start_deg, end=end_deg, fill=line_color, width=line_width)

        safe_print(f"[OK] Cross at {angle:.2f} deg (0 deg at top, clockwise) -> draw_angle={draw_angle_deg:.2f} deg")

    # --- 两角度完整弧（顺时针从 start→end） ---
    if connect_arc and len(angle_list) == 2:
        start_angle, end_angle = angle_list  # 均为“0°在上、顺时针”

        start_draw = to_draw_angle(start_angle)
        end_draw   = to_draw_angle(end_angle)

        bbox = [cx - inner_radius, cy - inner_radius,
                cx + inner_radius, cy + inner_radius]

        # Pillow 的角度也是从右侧开始顺时针递增，
        # 我们已经把角度线性变换为同一体系：直接 start_draw → end_draw 即为顺时针弧。
        draw.arc(bbox, start=start_draw, end=end_draw,
                 fill=line_color, width=line_width)

        arc_len = (end_angle - start_angle) % 360
        safe_print(
            f"[ARC] CW {start_angle:.2f} deg -> {end_angle:.2f} deg "
            f"(draw {start_draw:.2f} deg -> {end_draw:.2f} deg, arc {arc_len:.2f} deg)"
        )

    img.save(output_path)
    safe_print(f"[SAVE] {output_path}")
    return output_path

def _sanitize_filename(filename: str) -> str:
    """替换掉文件名里不合法的字符，避免路径报错"""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)

def crop_sector_for_amplifier(
        image_path: str,
        centre: tuple[int, int],
        inner_r: int,
        outer_r: int,
        feedback_angles: dict,
        chart_id: str,
        point_name: str,
        pad_start_ccw: float = 30.0,
        pad_end_cw: float = 30.0,
        zoom_scale: float = 3.0,
        grid_interval_deg: int = 5,
        grid_line_ratio: float = 0.10,
        grid_line_inward_ratio: float = 0.03,
        text_offset_ratio: float = 0.12,
        out_bg=(255, 255, 255, 255),
        save_suffix: str = "",
        amp_round: int = 1,
        feedback_round: int = 0,
) -> tuple[str, list[int], str | None]:
    """
    完整升级后的 Amplifier 扇区裁剪函数（0° 在上方，顺时针递增角度体系）。
    - 完全保持 Option A 原视觉效果
    - 所有角度均已统一到新体系（0°=上，90°=右，180°=下，270°=左）
    - 完全兼容 draw_angle_grid_30deg / draw_angle_feedback
    - 保留原所有 amplifier 行为（pad、LLM 包含判定、swap、LLM 顺序判断…）
    """

    import os, math, json, re
    from PIL import Image, ImageDraw, ImageFont

    # ----------------------------------------------------------------------
    # ⭐（0）LLM 工具：原样保留
    # ----------------------------------------------------------------------

    def _run_coro_in_thread(coro):
        import threading, asyncio
        box = {"res": None, "err": None}

        def _worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                box["res"] = loop.run_until_complete(coro)
            except Exception as e:
                box["err"] = e
            finally:
                loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        if box["err"] is not None:
            raise box["err"]
        return box["res"]

    def _deep_find(obj, key_lower: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() == key_lower:
                    return v
                result = _deep_find(v, key_lower)
                if result is not None:
                    return result
        elif isinstance(obj, (list, tuple)):
            for it in obj:
                result = _deep_find(it, key_lower)
                if result is not None:
                    return result
        elif isinstance(obj, str):
            try:
                j = json.loads(obj)
                return _deep_find(j, key_lower)
            except:
                return None
        return None

    def _parse_llm_contains(resp):
        import re
        if resp is None:
            return None
        txt = str(resp)
        txt = re.sub(r'^```json', '', txt)
        txt = re.sub(r'```$', '', txt).strip()
        try:
            j = json.loads(txt)
            val = _deep_find(j, 'contains')
            if isinstance(val, bool):
                return val
        except:
            pass
        if re.search(r'\btrue\b', txt, re.I):
            return True
        if re.search(r'\bfalse\b', txt, re.I):
            return False
        return None

    def _llm_validate_contains(image_path_, item_name_):
        return None
        import asyncio
        if "call_llm_once" not in globals():
            return None
        prompt = (
            "You are given a cropped donut/pie chart sector.\n"
            f"Determine whether the ENTIRE sector \"{item_name_}\" is fully included.\n"
            "Return JSON: {\"contains\": true/false}"
        )
        try:
            try:
                asyncio.get_running_loop()
                resp = _run_coro_in_thread(globals()["call_llm_once"](prompt, image_path_))
            except RuntimeError:
                resp = asyncio.run(globals()["call_llm_once"](prompt, image_path_))
            return _parse_llm_contains(resp)
        except Exception:
            return None

    # def _parse_llm_cross_zero(resp):
    #     import re
    #     if resp is None:
    #         return None
    #     txt = str(resp)
    #     txt = re.sub(r'^```json', '', txt)
    #     txt = re.sub(r'```$', '', txt).strip()
    #     try:
    #         j = json.loads(txt)
    #         val = _deep_find(j, 'crosses_zero')
    #         if isinstance(val, bool):
    #             return val
    #     except:
    #         pass
    #     if re.search(r'\btrue\b', txt, re.I):
    #         return True
    #     if re.search(r'\bfalse\b', txt, re.I):
    #         return False
    #     return None
    #
    # def _llm_validate_cross_zero(image_path_, item_name_):
    #     import asyncio
    #     if "call_llm_once" not in globals():
    #         return None
    #
    #     prompt = (
    #         "You are given a full 360° donut / pie chart crop.\n"
    #         "The red radial line marks the exact 0° reference angle (pointing upward).\n"
    #         f"Determine whether the sector \"{item_name_}\" crosses the 0° boundary.\n"
    #         "A sector crosses 0° if any part appears on both sides of the vertical line.\n"
    #         "Return JSON: {\"crosses_zero\": true/false}"
    #     )
    #
    #     try:
    #         try:
    #             asyncio.get_running_loop()
    #             resp = _run_coro_in_thread(globals()["call_llm_once"](prompt, image_path_))
    #         except RuntimeError:
    #             resp = asyncio.run(globals()["call_llm_once"](prompt, image_path_))
    #         return _parse_llm_cross_zero(resp)
    #     except:
    #         return None

    # ----------------------------------------------------------------------
    # ⭐（1）Amplifier 自动配置
    # ----------------------------------------------------------------------
    amp_cfg = {
        1: {"pad": 22.0, "grid": 5, "zoom": 2.4},
        2: {"pad": 14.0, "grid": 3, "zoom": 2.6},
        3: {"pad": 9.0, "grid": 2, "zoom": 3.0},
    }
    cfg = amp_cfg.get(amp_round, amp_cfg[3])
    pad_start_ccw = cfg["pad"]
    pad_end_cw = cfg["pad"]
    grid_interval_deg = cfg["grid"]
    zoom_scale = cfg["zoom"]

    # ----------------------------------------------------------------------
    # ⭐（2）角度体系 —— 新版标准化（核心）
    # ----------------------------------------------------------------------
    # 所有 angle_deg 均为：
    #   0° = 上
    #   顺时针递增
    # 将其转换为用于绘制（Pillow arc 0° 在右）
    def to_draw_angle(angle):
        return (angle - 90) % 360

    # ----------------------------------------------------------------------
    # ⭐（3）角度扩展（按顺时针）
    # ----------------------------------------------------------------------
    start = float(feedback_angles["start_angle"])
    end   = float(feedback_angles["end_angle"])
    arc_len = (end - start) % 360

    if arc_len >= 330:  # 全圆
        s_ext = start % 360
        e_ext = (start + arc_len) % 360
    else:
        s_ext = (start - pad_start_ccw) % 360
        e_ext = (end + pad_end_cw) % 360
        if (e_ext - s_ext) % 360 >= 359.9:
            s_ext, e_ext = 0, 360

    # ----------------------------------------------------------------------
    # ⭐（4）生成 mask 扇区（按新角度体系）
    # ----------------------------------------------------------------------
    img = Image.open(image_path).convert("RGBA")
    W, H = img.size
    cx, cy = centre

    mask = Image.new("L", (W, H), 0)
    dm = ImageDraw.Draw(mask)

    # 外圆 bbox
    outer_bbox = [cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r]

    if arc_len >= 330:  # 全圆
        dm.ellipse(outer_bbox, fill=255)
    else:
        pts = [(cx, cy)]

        raw = (e_ext - s_ext) % 360
        steps = max(1, int(math.ceil(raw)))

        for i in range(steps + 1):
            a = (s_ext + raw * (i / steps)) % 360
            th = math.radians((a - 90) % 360)
            x = cx + outer_r * math.cos(th)
            y = cy + outer_r * math.sin(th)
            pts.append((x, y))

        dm.polygon(pts, fill=255)

    # ----------------------------------------------------------------------
    # ⭐（5）mask 裁剪 + 初次 zoom
    # ----------------------------------------------------------------------
    arc_img = Image.new("RGBA", (W, H), out_bg)
    arc_img.paste(img, (0, 0), mask=mask)

    bbox = mask.getbbox()
    if not bbox:
        raise ValueError("Empty mask")

    extra_pad = int(outer_r * 0.20)
    lx = max(bbox[0] - extra_pad, 0)
    ly = max(bbox[1] - extra_pad, 0)
    rx = min(bbox[2] + extra_pad, W)
    by = min(bbox[3] + extra_pad, H)

    cropped = arc_img.crop((lx, ly, rx, by))

    zoomed = cropped.resize(
        (int(cropped.width * zoom_scale), int(cropped.height * zoom_scale)),
        Image.BICUBIC
    )

    # ----------------------------------------------------------------------
    # ⭐（6）zoom 后更新圆心
    # ----------------------------------------------------------------------
    orig_left = cx - outer_r
    orig_top  = cy - outer_r

    crop_offset_x = orig_left - lx
    crop_offset_y = orig_top - ly

    cx_new = int((crop_offset_x + outer_r) * zoom_scale)
    cy_new = int((crop_offset_y + outer_r) * zoom_scale)

    # ----------------------------------------------------------------------
    # ⭐（7）若 zoom 过小 → 强制放大
    # ----------------------------------------------------------------------
    MIN_TARGET = amplifier_target_side(family="sector")
    zw, zh = zoomed.size
    ss = min(zw, zh)

    if ss < MIN_TARGET:
        scale2 = MIN_TARGET / ss
        zoomed = zoomed.resize(
            (int(zw * scale2), int(zh * scale2)),
            Image.BICUBIC
        )
        cx_new = int(cx_new * scale2)
        cy_new = int(cy_new * scale2)
    long_side = max(zoomed.size)
    if long_side > amplifier_max_side():
        scale3 = amplifier_max_side() / long_side
        zoomed = zoomed.resize(
            (max(1, int(zoomed.width * scale3)), max(1, int(zoomed.height * scale3))),
            Image.BICUBIC,
        )
        cx_new = int(cx_new * scale3)
        cy_new = int(cy_new * scale3)

    # ----------------------------------------------------------------------
    # ⭐（8）扫描真实半径（新版）
    # ----------------------------------------------------------------------
    zw, zh = zoomed.size

    def scan_radius(a):
        th = math.radians((a - 90) % 360)
        dx, dy = math.cos(th), math.sin(th)
        max_r = int(math.hypot(zw, zh))
        for r in range(max_r, 0, -1):
            x = int(cx_new + dx * r)
            y = int(cy_new + dy * r)
            if 0 <= x < zw and 0 <= y < zh:
                if zoomed.getpixel((x, y))[:3] != (255, 255, 255):
                    return r
        return 0

    raw_samples = [scan_radius(a) for a in range(0, 360, 30)]
    valid = [x for x in raw_samples if x > 0]

    if not valid:
        r_edge_global = min(zw, zh) // 2
    else:
        mx = max(valid)
        filtered = sorted([v for v in valid if v >= mx * 0.5])
        n = len(filtered)
        if n >= 5:
            k = max(1, int(n * 0.2))
            mid = filtered[k:-k] or filtered
        else:
            mid = filtered
        m = len(mid)
        if m % 2 == 1:
            r_edge_global = mid[m // 2]
        else:
            r_edge_global = (mid[m // 2 - 1] + mid[m // 2]) // 2

    # ----------------------------------------------------------------------
    # ⭐（9）自动字体大小（与原版策略兼容）
    # ----------------------------------------------------------------------
    img_w, img_h = zoomed.size
    short_side = min(img_w, img_h)

    if short_side < 800:
        img_factor = 0.5
    elif short_side < 1400:
        img_factor = 0.7
    elif short_side < 2000:
        img_factor = 0.9
    else:
        img_factor = 1.8

    target_scale = 0.05
    base_size = r_edge_global * target_scale

    arc_len_local = 2 * math.pi * 0.6 * r_edge_global * (grid_interval_deg / 360)
    max_no_overlap = (arc_len_local * 0.85) / 0.6

    font_size = int(min(base_size * img_factor, max_no_overlap))
    font_size = max(14, min(font_size, 100))

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # ----------------------------------------------------------------------
    # ⭐（10）绘制 Overlay（刻度线 + 数字 + 两端边界线）
    # ----------------------------------------------------------------------
    overlay = Image.new("RGBA", zoomed.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(overlay)

    # 边界线（start/end）
    def draw_boundary(angle, color=(140, 140, 140, 255), width=6):
        th = math.radians((angle - 90) % 360)
        dx, dy = math.cos(th), math.sin(th)
        r_out = max(int(r_edge_global * 1.05), int(short_side * 0.25))
        x1 = cx_new + dx * r_out
        y1 = cy_new + dy * r_out
        dr.line([(cx_new, cy_new), (x1, y1)], fill=color, width=width)

    draw_boundary(s_ext)
    draw_boundary(e_ext)

    # 刻度绘制区间（顺时针）
    angle_start = s_ext
    angle_end   = e_ext
    if angle_end < angle_start:
        angle_end += 360

    def in_range(a):
        aa = a
        if aa < angle_start:
            aa += 360
        return angle_start <= aa <= angle_end

    drawn_angles = []
    zero_label_info = None  # ⭐ 用来记录 0° 的 (tx, ty, rotate_angle)

    for base in range(0, 720, grid_interval_deg):
        a = base % 360

        # ⭐ 0° 特例：无论 in_range 与否，都强制绘制
        if a != 0 and not in_range(base):
            continue

        drawn_angles.append(a)

        # r_edge = scan_radius(a)
        r_edge = r_edge_global

        r0 = max(5, r_edge - int(r_edge * grid_line_inward_ratio))
        r1 = r_edge + int(r_edge * grid_line_ratio)

        th = math.radians((a - 90) % 360)
        dx, dy = math.cos(th), math.sin(th)

        x0 = cx_new + dx * r0
        y0 = cy_new + dy * r0
        x1 = cx_new + dx * r1
        y1 = cy_new + dy * r1

        if a == 0:
            dr.line([(x0, y0), (x1, y1)], fill=(255, 0, 0, 255), width=5)
        else:
            dr.line([(x0, y0), (x1, y1)], fill=(0, 0, 0, 255), width=1)

            # =========================
            #  文字：沿当前径向线方向
            # =========================
            label = f"{a}°"

            # 文字中心：沿着刚才的径向线再往外一点
            r_text = r1 + int(r_edge_global * 0.06)
            tx = cx_new + dx * r_text
            ty = cy_new + dy * r_text

            # 用 font.getbbox 拿文字尺寸
            bbox_t = font.getbbox(label)
            tw = bbox_t[2] - bbox_t[0]
            thh = bbox_t[3] - bbox_t[1]

            # 多留一点 padding，避免旋转时被边缘“吃掉”
            pad = max(6, font_size // 3)
            box_w = tw + 2 * pad
            box_h = thh + 2 * pad

            # 1) 在小画布上画“白底 + 黑字”
            text_box = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
            td = ImageDraw.Draw(text_box)

            # 白底保持不变：纯白、不透明
            td.rectangle([0, 0, box_w, box_h], fill=(255, 255, 255, 255))

            # 文本往里缩 pad 像素，保证不贴边
            td.text(
                (pad, pad),
                label,
                font=font,
                fill=(0, 0, 0, 255),
                stroke_width=1,  # 细描边，避免边缘被插值吃掉
                stroke_fill=(0, 0, 0, 255),
            )

            # =====================================================
            # 旋转角度换算：
            #   a 是「0° 在正上，顺时针为正」
            #   Pillow.rotate(angle) 使用「0° 在正右，逆时针为正」
            # 两者转换关系：
            #   image_angle = (90 - a) % 360
            #
            # 举例：
            #   a =   0°（正上） → image =  90°（从右逆时针到上）
            #   a =  90°（正右） → image =   0°
            #   a = 180°（正下） → image = 270°
            #   a = 270°（正左） → image = 180°
            #   a = 330°        → image = 120°  ✅ 正是你说的
            # =====================================================
            # =====================================================
            # 旋转角度换算（分上下半圈）：
            #   a 是「0° 在正上，顺时针为正」
            #   Pillow.rotate(angle) 使用「0° 在正右，逆时针为正」
            #   基础映射：base_angle = (90 - a) % 360
            #   - 0°~180°：从里往外读 → 不翻转
            #   - 180°~360°：从外往里读 → 再加 180°
            # =====================================================
            base_angle = (90.0 - a) % 360.0

            if 0.0 <= a <= 180.0:
                rotate_angle = base_angle  # 上半圈：里 → 外
            else:
                rotate_angle = (base_angle + 180.0) % 360.0  # 下半圈：外 → 里

            # ⭐ 如果是 0°，顺手把它的中心位置 & 旋转角度记下来
            if a == 0:
                zero_label_info = (tx, ty, rotate_angle)

            rotated = text_box.rotate(rotate_angle, expand=True)

            rw, rh = rotated.size

            # 3) 按中心对齐，把旋转后的文字贴回 overlay
            overlay.paste(rotated, (int(tx - rw / 2), int(ty - rh / 2)), rotated)

        # =========================================================
        # ⭐ 循环结束后，把 0° 再画一遍，保证在最上面、一定可见
        # =========================================================
        if zero_label_info is not None:
            tx0, ty0, rot0 = zero_label_info
            label0 = "0°"

            bbox0 = font.getbbox(label0)
            tw0 = bbox0[2] - bbox0[0]
            th0 = bbox0[3] - bbox0[1]

            pad0 = max(6, font_size // 3)
            box_w0 = tw0 + 2 * pad0
            box_h0 = th0 + 2 * pad0

            text_box0 = Image.new("RGBA", (box_w0, box_h0), (0, 0, 0, 0))
            d0 = ImageDraw.Draw(text_box0)
            d0.rectangle([0, 0, box_w0, box_h0], fill=(255, 255, 255, 255))
            d0.text(
                (pad0, pad0),
                label0,
                font=font,
                fill=(0, 0, 0, 255),
                stroke_width=1,
                stroke_fill=(0, 0, 0, 255),
            )

            rotated0 = text_box0.rotate(rot0, expand=True)
            rw0, rh0 = rotated0.size

            overlay.paste(
                rotated0,
                (int(tx0 - rw0 / 2), int(ty0 - rh0 / 2)),
                rotated0,
            )

            # ============================
            # ⭐ 强制补画一个 0° 标签
            # ============================

        # =========================================================
        # ⭐ 修复后的 0° 补画逻辑 — 仅当 0° 在扇区区间内才绘制
        # =========================================================

        # 判断 0° 是否在有效区间
        zero_in_range = in_range(0)

        if zero_in_range:
            a0 = 0
            label0 = "0°"

            # 全局半径（无需扫描）
            r_edge0 = r_edge_global

            # tick 半径
            r0_0 = r_edge0 - int(r_edge0 * grid_line_inward_ratio)
            r1_0 = r_edge0 + int(r_edge0 * grid_line_ratio)
            r_text0 = r1_0 + int(r_edge_global * 0.06)

            # 0° = 正上方方向 (dx=0, dy=-1)
            dx0, dy0 = 0.0, -1.0

            tx0 = cx_new + dx0 * r_text0
            ty0 = cy_new + dy0 * r_text0

            # 文字尺寸
            bbox0 = font.getbbox(label0)
            tw0 = bbox0[2] - bbox0[0]
            th0 = bbox0[3] - bbox0[1]

            pad0 = max(6, font_size // 3)
            box_w0 = tw0 + 2 * pad0
            box_h0 = th0 + 2 * pad0

            text_box0 = Image.new("RGBA", (box_w0, box_h0), (0, 0, 0, 0))
            d0 = ImageDraw.Draw(text_box0)
            d0.rectangle([0, 0, box_w0, box_h0], fill=(255, 255, 255, 255))
            d0.text((pad0, pad0), label0, font=font, fill=(0, 0, 0, 255),
                    stroke_width=1, stroke_fill=(0, 0, 0))

            # 旋转（0° 属于翻转区）
            angle_img0 = math.degrees(math.atan2(dy0, dx0))
            rotate_angle0 = (angle_img0 + 180) % 360

            rot0 = text_box0.rotate(rotate_angle0, expand=True)
            rw0, rh0 = rot0.size

            overlay.paste(rot0, (int(tx0 - rw0 / 2), int(ty0 - rh0 / 2)), rot0)
        # else: 不绘制 0°

        # =========================================================
        # ⭐ 强制补画 0° 标签 —— 半径、方向完全对齐主循环
        # =========================================================
        # a0 = 0
        # label0 = "0°"
        #
        # # 1) 半径扫描（若异常用全局半径）
        # r_edge0 = r_edge_global
        # # r_edge0 = scan_radius(a0)
        # # if r_edge0 <= 0:
        # #     r_edge0 = r_edge_global
        #
        # # 2) 刻度线与文本半径（与主循环严格一致）
        # r0_0 = r_edge0 - int(r_edge0 * grid_line_inward_ratio)
        # r1_0 = r_edge0 + int(r_edge0 * grid_line_ratio)
        # r_text0 = r1_0 + int(r_edge_global * 0.06)
        #
        # # 3) 正上方方向（0°：dx=0, dy=-1）
        # dx0, dy0 = 0.0, -1.0
        # tx0 = cx_new + dx0 * r_text0
        # ty0 = cy_new + dy0 * r_text0
        #
        # # 4) 文本贴图（白底 + 黑字，与主循环完全一致）
        # bbox0 = font.getbbox(label0)
        # tw0 = bbox0[2] - bbox0[0]
        # th0 = bbox0[3] - bbox0[1]
        # pad0 = max(6, font_size // 3)
        # box_w0 = tw0 + 2 * pad0
        # box_h0 = th0 + 2 * pad0
        #
        # text_box0 = Image.new("RGBA", (box_w0, box_h0), (0, 0, 0, 0))
        # d0 = ImageDraw.Draw(text_box0)
        # d0.rectangle([0, 0, box_w0, box_h0], fill=(255, 255, 255, 255))
        # d0.text((pad0, pad0), label0, font=font, fill=(0, 0, 0, 255),
        #         stroke_width=1, stroke_fill=(0, 0, 0))
        #
        # # 5) 旋转（沿径向线 + 0° 属于 0–180° → 内→外 需再翻转 180°）
        # angle_img0 = math.degrees(math.atan2(dy0, dx0))
        # rotate_angle0 = (angle_img0 + 180) % 360  # 0° 总是属于翻转区
        #
        # rot0 = text_box0.rotate(rotate_angle0, expand=True)
        # rw0, rh0 = rot0.size
        #
        # # 6) 按中心贴入 overlay（保证在其它内容上方）
        # overlay.paste(rot0, (int(tx0 - rw0 / 2), int(ty0 - rh0 / 2)), rot0)



    final_img = Image.alpha_composite(zoomed, overlay)

    # ----------------------------------------------------------------------
    # ⭐（11）保存
    # ----------------------------------------------------------------------
    output_root = AMPLIFIER_OUTPUT_ROOT or "results_Pixtral"
    out_dir = os.path.join(output_root, chart_id, "amplifier_img")
    os.makedirs(out_dir, exist_ok=True)

    safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", point_name)

    if "_swap" in save_suffix:
        filename = f"{safe_name}_amp{amp_round}_swap.png"
    else:
        filename = f"{safe_name}_amp{amp_round}.png"

    out_path = os.path.join(out_dir, filename)
    final_img.save(out_path)

    # ----------------------------------------------------------------------
    # ⭐（12）LLM 顺序判断（保持原逻辑）
    # ----------------------------------------------------------------------
    arc_for_draw = (e_ext - s_ext) % 360
    is_full = arc_len >= 330 or arc_for_draw >= 359.9 or abs(arc_for_draw) <= 1e-6
    hint = None

    try:
        prev_s = feedback_angles["start_angle"]
        prev_e = feedback_angles["end_angle"]

        if is_full:
            cross = _llm_validate_cross_zero(out_path, point_name)
            if cross is True:
                if prev_s <= prev_e:
                    hint = (
                        f"The sector crosses the 0° line, "
                        f"so start/end should be swapped: "
                        f"start≈{prev_e}, end≈{prev_s}"
                    )
                else:
                    hint = (
                        f"The sector crosses 0°, keep order: start > end."
                    )
            elif cross is False:
                if prev_s <= prev_e:
                    hint = "Sector does not cross, keep start<end"
                else:
                    hint = (
                        f"Sector does not cross, order is incorrect. "
                        f"The correct is that start≈{prev_e}, end≈{prev_s}"
                    )
        else:
            contains = _llm_validate_contains(out_path, point_name)
            if contains is True:
                hint = (
                    f"Keep same order: start≈{prev_s}, end≈{prev_e}"
                )
            elif contains is False:
                if prev_s <= prev_e:
                    hint = (
                        f"Order wrong: true start > end. "
                        f"start≈{prev_e}, end≈{prev_s}"
                    )
                else:
                    hint = (
                        f"Order wrong: true start < end. "
                        f"start≈{prev_e}, end≈{prev_s}"
                    )

                if "_swap" not in save_suffix:
                    swapped = {
                        "start_angle": prev_e,
                        "end_angle": prev_s
                    }
                    recrop_path, recrop_angles, _ = crop_sector_for_amplifier(
                        image_path=image_path,
                        centre=centre,
                        inner_r=inner_r,
                        outer_r=outer_r,
                        feedback_angles=swapped,
                        chart_id=chart_id,
                        point_name=point_name,
                        pad_start_ccw=pad_start_ccw,
                        pad_end_cw=pad_end_cw,
                        zoom_scale=zoom_scale,
                        grid_interval_deg=grid_interval_deg,
                        grid_line_ratio=grid_line_ratio,
                        grid_line_inward_ratio=grid_line_inward_ratio,
                        text_offset_ratio=text_offset_ratio,
                        out_bg=out_bg,
                        save_suffix=save_suffix + "_swap",
                        amp_round=amp_round,
                        feedback_round=feedback_round,
                    )
                    return recrop_path, recrop_angles, hint

    except Exception as e:
        safe_print("[LLM] Validation skipped:", e)

    return out_path, drawn_angles, hint


    import os, math, json, re
    from PIL import Image, ImageDraw, ImageFont

    # =====================================================
    # ⭐ 0. 注入：LLM 工具（来自旧版本，原封不动）
    # =====================================================

    def _run_coro_in_thread(coro):
        import threading, asyncio
        box = {"res": None, "err": None}

        def _worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                box["res"] = loop.run_until_complete(coro)
            except Exception as e:
                box["err"] = e
            finally:
                loop.close()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()
        if box["err"] is not None:
            raise box["err"]
        return box["res"]

    # ---- 深度 JSON 搜索（旧版保留） ----
    def _deep_find(obj, key_lower: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() == key_lower:
                    return v
                found = _deep_find(v, key_lower)
                if found is not None:
                    return found
        elif isinstance(obj, (list, tuple)):
            for it in obj:
                found = _deep_find(it, key_lower)
                if found is not None:
                    return found
        elif isinstance(obj, str):
            try:
                j = json.loads(obj)
                return _deep_find(j, key_lower)
            except Exception:
                return None
        return None

    # ---- contains 判定（旧版原样） ----
    def _parse_llm_contains(resp):
        import re
        if resp is None:
            return None
        txt = str(resp)
        # 包含 markdown 清理
        txt = re.sub(r"^```json", "", txt)
        txt = re.sub(r"```$", "", txt).strip()

        try:
            j = json.loads(txt)
            val = _deep_find(j, "contains")
            if isinstance(val, bool):
                return val
        except:
            pass

        if re.search(r'\btrue\b', txt, re.I):
            return True
        if re.search(r'\bfalse\b', txt, re.I):
            return False
        return None

    def _llm_validate_contains(image_path_, item_name_):
        return None
        import asyncio
        if "call_llm_once" not in globals():
            return None
        prompt = (
            "You are given a cropped donut/pie chart sector.\n"
            f"Determine whether the ENTIRE sector \"{item_name_}\" is fully included.\n"
            "Return JSON: {\"contains\": true/false}"
        )
        try:
            try:
                asyncio.get_running_loop()
                resp = _run_coro_in_thread(globals()["call_llm_once"](prompt, image_path_))
            except RuntimeError:
                resp = asyncio.run(globals()["call_llm_once"](prompt, image_path_))
            return _parse_llm_contains(resp)
        except Exception:
            return None

    # # ---- cross_zero 判定（旧版原样） ----
    # def _parse_llm_cross_zero(resp):
    #     import re
    #     if resp is None:
    #         return None
    #     txt = str(resp)
    #     txt = re.sub(r"^```json", "", txt)
    #     txt = re.sub(r"```$", "", txt).strip()
    #     try:
    #         j = json.loads(txt)
    #         val = _deep_find(j, "crosses_zero")
    #         if isinstance(val, bool):
    #             return val
    #     except:
    #         pass
    #     if re.search(r'\btrue\b', txt, re.I):
    #         return True
    #     if re.search(r'\bfalse\b', txt, re.I):
    #         return False
    #     return None
    #
    # def _llm_validate_cross_zero(image_path_, item_name_):
    #     import asyncio
    #     if "call_llm_once" not in globals():
    #         return None
    #
    #     prompt = (
    #         f'You are given a full 360° donut / pie chart crop.'
    #         f'The red radial line marks the exact 0° reference angle and is located at the **positive horizontal direction (pointing to the right)**.'
    #         f'Your task is to determine whether the sector "{item_name_}" crosses the 0° boundary.'
    #
    #         '''
    #         Important:
    #         - Treat the 0° line as a strict boundary.
    #         - A sector is considered to “cross 0°” if any part of it appears on BOTH sides of the red radial line.
    #         - This includes even minimal overlap: if the sector touches or slightly spills over the 0° line by even 1° or a visually tiny amount, count it as crossing.
    #         - First, identify the clockwise and clockwise extents of the sector relative to the 0° line, then judge whether the span covers both sides.
    #
    #         Return JSON: {\"crosses_zero\": true/false}
    #         '''
    #     )
    #     # prompt = (
    #     #     "You are given a full 360° donut/pie chart crop.\n"
    #     #     "The red radial line marks 0°.\n"
    #     #     f"Determine whether sector \"{item_name_}\" crosses 0°.\n"
    #     #     "Return JSON: {\"crosses_zero\": true/false}"
    #     # )
    #     try:
    #         try:
    #             asyncio.get_running_loop()
    #             resp = _run_coro_in_thread(globals()["call_llm_once"](prompt, image_path_))
    #         except RuntimeError:
    #             resp = asyncio.run(globals()["call_llm_once"](prompt, image_path_))
    #         return _parse_llm_cross_zero(resp)
    #     except:
    #         return None

    # =====================================================
    # ① AMP 自动参数
    # =====================================================
    amp_cfg = {
        1: {"pad": 22.0, "grid": 5, "zoom": 2.4},
        2: {"pad": 14.0, "grid": 3, "zoom": 2.6},
        3: {"pad": 9.0, "grid": 2, "zoom": 3.0},
    }
    cfg = amp_cfg.get(amp_round, amp_cfg[3])
    pad_start_ccw = cfg["pad"]
    pad_end_cw = cfg["pad"]
    grid_interval_deg = cfg["grid"]
    zoom_scale = cfg["zoom"]

    # =====================================================
    # ② 角度扩展
    # =====================================================
    def to_pil_angle(a): return (-a) % 360

    s = float(feedback_angles["start_angle"])
    e = float(feedback_angles["end_angle"])
    arc = (e - s) % 360

    if arc >= 330:
        s_ext = s % 360
        e_ext = (s + arc) % 360
    else:
        s_ext = (s - pad_start_ccw) % 360
        e_ext = (e + pad_end_cw) % 360
        if (e_ext - s_ext) % 360 >= 359.9:
            s_ext, e_ext = 0, 360

    # =====================================================
    # ③ mask 扇区
    # =====================================================
    img = Image.open(image_path).convert("RGBA")
    W, H = img.size
    cx, cy = centre

    mask = Image.new("L", (W, H), 0)
    draw_mask = ImageDraw.Draw(mask)

    outer_bbox = [cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r]

    if arc >= 330:
        draw_mask.ellipse(outer_bbox, fill=255)
    else:
        pts = [(cx, cy)]

        # 正确的 CCW 弧长
        raw_diff = (e_ext - s_ext) % 360
        steps = int(max(1, math.ceil(raw_diff)))

        for i in range(steps + 1):
            frac = i / steps
            a = (s_ext + raw_diff * frac) % 360
            th = math.radians(360 - a)
            pts.append((cx + outer_r * math.cos(th), cy + outer_r * math.sin(th)))

        draw_mask.polygon(pts, fill=255)

    arc_img = Image.new("RGBA", (W, H), out_bg)
    arc_img.paste(img, (0, 0), mask=mask)

    bbox = mask.getbbox()
    if not bbox:
        raise ValueError("Mask empty")

    # =====================================================
    # ④ 裁剪 + 初次 zoom
    # =====================================================
    extra_pad = int(outer_r * 0.20)
    bbox2 = (
        max(bbox[0] - extra_pad, 0),
        max(bbox[1] - extra_pad, 0),
        min(bbox[2] + extra_pad, W),
        min(bbox[3] + extra_pad, H),
    )

    cropped = arc_img.crop(bbox2)
    cropped = Image.alpha_composite(Image.new("RGBA", cropped.size, (255, 255, 255, 255)), cropped)

    zoomed = cropped.resize(
        (int(cropped.width * zoom_scale), int(cropped.height * zoom_scale)),
        Image.BICUBIC
    )

    # =====================================================
    # ⑤ 精确圆心修复（随裁剪与 zoom 更新）
    # =====================================================
    orig_left = cx - outer_r
    orig_top = cy - outer_r

    crop_offset_x = orig_left - bbox2[0]
    crop_offset_y = orig_top - bbox2[1]

    cx_new = int((crop_offset_x + outer_r) * zoom_scale)
    cy_new = int((crop_offset_y + outer_r) * zoom_scale)

    # =====================================================
    # ⑤.5 若 zoom 后仍然太小 → 强制放大到目标尺寸
    # =====================================================
    MIN_TARGET_SIZE = amplifier_target_side(family="sector")

    zw, zh = zoomed.size
    short_side_after_zoom = min(zw, zh)

    if short_side_after_zoom < MIN_TARGET_SIZE:
        scale2 = MIN_TARGET_SIZE / short_side_after_zoom
        zoomed = zoomed.resize(
            (int(zw * scale2), int(zh * scale2)),
            Image.BICUBIC
        )
        # 圆心跟随放大
        cx_new = int(cx_new * scale2)
        cy_new = int(cy_new * scale2)
        safe_print(f"[AMP AUTO-UPSCALE] enlarged x{scale2:.2f} -> size={zoomed.size}")
    long_side = max(zoomed.size)
    if long_side > amplifier_max_side():
        scale3 = amplifier_max_side() / long_side
        zoomed = zoomed.resize(
            (max(1, int(zoomed.width * scale3)), max(1, int(zoomed.height * scale3))),
            Image.BICUBIC,
        )
        cx_new = int(cx_new * scale3)
        cy_new = int(cy_new * scale3)
        safe_print(f"[AMP AUTO-DOWNSCALE] capped x{scale3:.2f} -> size={zoomed.size}")

    # =====================================================
    # ⑥ 扫描真实半径（鲁棒版）
    # =====================================================
    zoom_w, zoom_h = zoomed.size

    def scan_radius(a_deg: float) -> int:
        """从圆心沿角度 a_deg 扫描，找到第一个非白像素的半径"""
        th = math.radians(a_deg)
        dx = math.cos(th)
        dy = -math.sin(th)
        max_r = int(math.hypot(zoom_w, zoom_h))
        for r in range(max_r, 0, -1):
            x = int(cx_new + dx * r)
            y = int(cy_new + dy * r)
            if 0 <= x < zoom_w and 0 <= y < zoom_h:
                if zoomed.getpixel((x, y))[:3] != (255, 255, 255):
                    return r
        return 0

    # ============================================================
    # 仅扫描一次全局半径：每10°扫描，速度快20~40倍
    # ============================================================
    raw_samples = [scan_radius(a) for a in range(0, 360, 15)]
    valid = [r for r in raw_samples if r > 0]

    if not valid:
        r_edge_global = min(zoom_w, zoom_h) // 2
    else:
        r_edge_global = int(sorted(valid)[len(valid) // 2])  # 中位数更稳定

    # 可选：略缩1~2%，保证不压到扇区
    r_edge_global = int(r_edge_global * 0.98)

    safe_print(f"[AMP] Using global radius = {r_edge_global}")

    # raw_samples = [scan_radius(a) for a in range(0, 360, 10)]
    # samples = [r for r in raw_samples if r > 0]
    #
    # if not samples:
    #     # 极端兜底：用画布短边的一半
    #     r_edge_global = min(zoom_w, zoom_h) // 2
    # else:
    #     r_max = max(samples)
    #     # 丢掉特别小的（小于最大值 50%）
    #     filtered = [r for r in samples if r >= 0.5 * r_max]
    #     if not filtered:
    #         filtered = samples
    #     filtered.sort()
    #     n = len(filtered)
    #     if n >= 5:
    #         k = max(1, int(n * 0.2))
    #         core = filtered[k:-k] or filtered
    #     else:
    #         core = filtered
    #     m = len(core)
    #     if m % 2 == 1:
    #         r_edge_global = core[m // 2]
    #     else:
    #         r_edge_global = int(0.5 * (core[m // 2 - 1] + core[m // 2]))
    #
    # r_edge_global = int(r_edge_global)
    # print("[AMP] r_edge_global =", r_edge_global,
    #       "| raw_min/max =", (min(raw_samples), max(raw_samples)))

    # ============================================================
    # ⑦ 字体自动调整
    # ============================================================

    # 用于重叠判断的角度集合（先算好）
    pre_angles = []
    angle_start = s_ext
    angle_end = e_ext
    angle_end_adj = angle_end if angle_end >= angle_start else angle_end + 360

    for base_angle in range(0, 720, grid_interval_deg):
        logical_angle = base_angle
        a_mod = base_angle % 360
        in_r = (
            arc >= 330 or
            (angle_start <= angle_end_adj and angle_start <= logical_angle <= angle_end_adj) or
            (angle_start > angle_end_adj and (logical_angle >= angle_start or logical_angle <= angle_end_adj))
        )
        if in_r:
            pre_angles.append(a_mod)

    img_w, img_h = zoomed.size
    short_side = min(img_w, img_h)

    # ⚙️ 1. 整体缩一点：img_factor 稍微收紧
    if short_side < 800:
        img_factor = 0.5      # 原来 0.7
    elif short_side < 1400:
        img_factor = 0.7      # 原来 1.5
    elif short_side < 2000:
        img_factor = 0.9      # 原来 2.0
    else:
        img_factor = 1.8      # 原来 2.0

    # ⚙️ 2. 基于半径的比例略降
    target_scale = 0.05       # 原来 0.06
    base_size = r_edge_global * target_scale

    # ⚙️ 3. 保持重叠约束不变
    arc_len = 2 * math.pi * 0.6 * r_edge_global * (grid_interval_deg / 360)
    max_no_overlap_size = (arc_len * 0.85) / 0.6

    # ⚙️ 4. 上限略压一点，避免夸张的大字
    font_size = int(min(base_size * img_factor, max_no_overlap_size))
    font_size = max(14, min(font_size, 100))  # 原来 min(..., 130)，下限 28→24 也略微放小点

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    safe_print("[AMP TRUE FINAL FONT SIZE] =", font_size,
          "| short_side =", short_side,
          "| r_edge_global =", r_edge_global)


    # =====================================================
    # ⑧ 绘制 overlay（网格 + 边界线 + 文字）
    # =====================================================
    overlay = Image.new("RGBA", zoomed.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 固定两端径向边界线：从圆心到 donut 外半径附近（灰色）
    def draw_fixed_boundary(angle_deg, color=(140, 140, 140, 255), width=6):
        th = math.radians(angle_deg)
        dx = math.cos(th)
        dy = -math.sin(th)
        # 线段长度：略超出外半径 + 提供一个最小可见长度下限
        r_outer = max(
            int(r_edge_global * 1.05),
            int(short_side * 0.25),
        )
        x0, y0 = cx_new, cy_new
        x1 = cx_new + dx * r_outer
        y1 = cy_new + dy * r_outer
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)

    # 左右端边界线（扩展后的 start / end）
    draw_fixed_boundary(s_ext)
    draw_fixed_boundary(e_ext)

    # 重新设定用于 in_range 的角度区间
    angle_start = s_ext
    angle_end = e_ext
    if angle_end < angle_start:
        angle_end += 360

    def in_range(a):
        if arc >= 330:
            return True
        if angle_start <= angle_end:
            return angle_start <= a <= angle_end
        return a >= angle_start or a <= angle_end

    drawn_angles = []

    for base in range(0, 720, grid_interval_deg):
        a = base % 360
        if not in_range(base):
            continue
        drawn_angles.append(a)

        # r_edge = scan_radius(a)
        r_edge = r_edge_global

        r0 = r_edge - int(r_edge * grid_line_inward_ratio)
        r1 = r_edge + int(r_edge * grid_line_ratio)

        th = math.radians(a)
        dx = math.cos(th)
        dy = -math.sin(th)

        x0 = cx_new + dx * r0
        y0 = cy_new + dy * r0
        x1 = cx_new + dx * r1
        y1 = cy_new + dy * r1

        draw.line([(x0, y0), (x1, y1)], fill=(0, 0, 0, 255), width=1)
        if a == 0:
            draw.line([(x0, y0), (x1, y1)], fill=(255, 0, 0, 255), width=5)

        # ============================================================
        #  PATCH: 正确沿径向方向倾斜文字（匹配用户图例）
        # ============================================================

        # ==== 文字 ====
        label = f"{a}°"

        # 1) 计算文字尺寸
        bbox = font.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # 2) 文字放置位置（沿径向）
        r_text = r1 + int(r_edge_global * 0.06)
        tx = cx_new + dx * r_text
        ty = cy_new + dy * r_text

        # 3) 创建纯文字图层（无白底）
        text_img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        td = ImageDraw.Draw(text_img)
        td.text((0, 0), label, font=font, fill=(0, 0, 0, 255))

        # 4) 旋转文本 —— 直接沿着角度方向旋转
        rotate_angle = a           # ⭐ 你的角度体系下这是绝对正确的
        rotated = text_img.rotate(rotate_angle, expand=True)

        # 5) 给旋转后的文字加紧贴白底（不会误导方向）
        rw, rh = rotated.size
        box = Image.new("RGBA", (rw + 8, rh + 8), (255, 255, 255, 255))
        box.paste(rotated, (4, 4), rotated)

        # 6) 放置到 overlay
        overlay.paste(box, (int(tx - (rw + 8) / 2), int(ty - (rh + 8) / 2)), box)


    final_img = Image.alpha_composite(zoomed, overlay)

    # =====================================================
    # ⑨ 保存文件
    # =====================================================
    out_dir = os.path.join("results_Qwen", chart_id, "amplifier_img")
    os.makedirs(out_dir, exist_ok=True)

    safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", point_name)

    if "_swap" in save_suffix:
        filename = f"{safe_name}_amp{amp_round}_swap.png"
    else:
        filename = f"{safe_name}_amp{amp_round}.png"

    out_path = os.path.join(out_dir, filename)
    final_img.save(out_path)
    safe_print(f"[SAVE FIXED] {out_path}")

    arc_for_draw = (e_ext - s_ext) % 360.0
    is_full_circle = (arc >= 330.0) or (arc_for_draw >= 359.9) or (abs(arc_for_draw) <= 1e-6)

    # =====================================================
    # ⑩ LLM 顺序判断 + 自动 swap（保持你原来的逻辑）
    # =====================================================
    angle_order_hint = None
    try:
        prev_start = feedback_angles["start_angle"]
        prev_end = feedback_angles["end_angle"]

        if is_full_circle:
            crosses_zero = _llm_validate_cross_zero(out_path, point_name)
            if crosses_zero is True:
                if prev_start <= prev_end:
                    angle_order_hint = (
                        f"The sector crosses the 0° line, so the prediction for start/end order "
                        f"was incorrect: the true start_angle should be around {prev_end:.1f}, "
                        f"and the end_angle should be around {prev_start:.1f}."
                    )
                else:
                    angle_order_hint = (
                        "The sector crosses the 0° line, you must keep the start and end angles "
                        "in the SAME order as the above previous prediction that is start_angle > end_angle."
                    )
            elif crosses_zero is False:
                if prev_start <= prev_end:
                    angle_order_hint = (
                        "The sector does not cross the 0° line, you must keep the start and end angles "
                        "in the SAME order as the above previous prediction that is start_angle < end_angle."
                    )
                else:
                    angle_order_hint = (
                        f"The sector does not cross the 0° line, so the prediction for start/end order "
                        f"was incorrect: the true start_angle should be around {prev_end:.1f}, "
                        f"and the end_angle should be around {prev_start:.1f}."
                    )

        else:
            contains_current = _llm_validate_contains(out_path, point_name)

            if contains_current is True:
                angle_order_hint = (
                    "You must keep the start and end angles in the SAME order as the above previous "
                    f"prediction that is the true start_angle should be around {prev_start:.1f} and the end_angle should be around {prev_end:.1f}."
                )

            elif contains_current is False:
                if prev_start <= prev_end:
                    angle_order_hint = (
                        f"The previous start/end order was incorrect, you must follow the true order "
                        f"that is start_angle > end_angle. The start_angle should be around {prev_end:.1f} "
                        f"while the end_angle should be around {prev_start:.1f}."
                    )
                else:
                    angle_order_hint = (
                        f"The previous start/end order was incorrect, you must follow the true order "
                        f"that is start_angle < end_angle. The start_angle should be around {prev_end:.1f} "
                        f"while the end_angle should be around {prev_start:.1f}."
                    )

                if "_swap" not in save_suffix:
                    swapped_feedback = {
                        "start_angle": feedback_angles["end_angle"],
                        "end_angle": feedback_angles["start_angle"],
                    }
                    recrop_out_path, recrop_drawn_angles, _ = crop_sector_for_amplifier(
                        image_path=image_path,
                        centre=centre,
                        inner_r=inner_r,
                        outer_r=outer_r,
                        feedback_angles=swapped_feedback,
                        chart_id=chart_id,
                        point_name=point_name,
                        pad_start_ccw=pad_start_ccw,
                        pad_end_cw=pad_end_cw,
                        zoom_scale=zoom_scale,
                        grid_interval_deg=grid_interval_deg,
                        grid_line_ratio=grid_line_ratio,
                        grid_line_inward_ratio=grid_line_inward_ratio,
                        text_offset_ratio=text_offset_ratio,
                        out_bg=out_bg,
                        save_suffix=save_suffix + "_swap",
                        amp_round=amp_round,
                        feedback_round=feedback_round,
                    )
                    return recrop_out_path, recrop_drawn_angles, angle_order_hint

    except Exception as _e:
        safe_print(f"[LLM] post-crop validation skipped due to error: {_e}")

    return out_path, drawn_angles, angle_order_hint
