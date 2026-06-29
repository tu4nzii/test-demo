"""直角系图表网格加密主流程。

本文件是 backend 中直角坐标系图表的核心处理编排层，负责把上传图像
转换为基础网格图、加密网格图和系统生成 JSON 所需的 tick/像素元数据。

当前主链路覆盖：
- v_bar / h_bar / v_stacked_bar / h_stacked_bar
- line
- scatter / bubble

外部调用入口通常是：
backend.main.process_chart_image
    -> type_detection.chart_processor.CartesianChartProcessor.encode_image
    -> process_chart(...)

实现原则：
1. 正常图表优先走 CV 检测结果，不主动补轴。
2. 只有上传阶段 MLLM 先验或 CV 失败表明缺轴/弱轴/仅网格时，才启用修复逻辑。
3. 只对数值轴插入加密 tick；文字轴只保留原始类别位置。
4. 系统处理过程不读取数据集 GT JSON。
"""

import os
import cv2
import numpy as np
import json
import logging
import sys
import asyncio
import argparse
import base64
import itertools
import math
import re
from glob import glob
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 后端运行目录与缓存目录。tick、颜色、柱端数值标签等 MLLM 结果统一缓存在 backend/data/llm_cache 下。
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_LOG_DIR = os.path.join(BACKEND_DIR, "data", "logs")
LLM_CACHE_DIR = os.path.join(BACKEND_DIR, "data", "llm_cache")
TICK_LABEL_CACHE_DIR = os.path.join(LLM_CACHE_DIR, "tick_labels")
BAR_VALUE_LABEL_CACHE_DIR = os.path.join(LLM_CACHE_DIR, "bar_value_labels")
COLOR_CACHE_DIR = os.path.join(LLM_CACHE_DIR, "colors")
ENCRYPTED_LABEL_STYLE_VERSION = "ocr_box_position_white_bg_axis_all_or_none_v1"

# 配置日志系统
log_path = os.path.join(RUNTIME_LOG_DIR, "grid_generation.log")
try:
    # 确保日志目录存在
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
except Exception as e:
    # 如果日志配置失败，至少确保有控制台输出
    print(f"警告: 无法配置日志文件，将只输出到控制台: {e}")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 添加backend目录到系统路径以导入统一配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========== 大模型API配置项（统一管理） ==========
# 模型 API 统一配置。这里不要写死 key/model，统一从 backend/model_api_config.py 读取。
from gemini_calls import FAILURE_TEXT, chat_with_gemini
from model_api_config import get_api_key, get_chat_completion_url, get_model_name

model_name = get_model_name()


def _env_flag_enabled(name, default=True):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _enhanced_reconstruction_outputs(output_root, image_stem, enhanced_mllm_cache_root, preview_path=None):
    out_base = Path(output_root) / image_stem
    selection_files = sorted(out_base.parent.glob(f"{out_base.name}_final*_selection.json"))
    final_grid_files = sorted(
        path
        for path in out_base.parent.glob(f"{out_base.name}_final*.png")
        if "overlay" not in path.name and "horizontal" not in path.name and "vertical" not in path.name
    )
    final_grid_path = final_grid_files[0] if final_grid_files else None
    final_overlay_path = (
        final_grid_path.with_name(final_grid_path.stem + "_overlay" + final_grid_path.suffix)
        if final_grid_path is not None
        else None
    )
    return {
        "preview_path": str(preview_path or out_base.with_name(out_base.name + "_preview.png")),
        "final_grid_path": str(final_grid_path) if final_grid_path is not None else "",
        "final_overlay_path": str(final_overlay_path) if final_overlay_path is not None and final_overlay_path.exists() else "",
        "final_selection_path": str(selection_files[0]) if selection_files else "",
        "axis_fusion_path": str(out_base.with_name(out_base.name + "_axis_fusion.json")),
        "grid_label_bindings_path": str(out_base.with_name(out_base.name + "_grid_label_bindings.json")),
        "grid_layers_path": str(out_base.with_name(out_base.name + "_grid_layers.json")),
        "mllm_axis_path": str(out_base.with_name(out_base.name + "_mllm_axis.json")),
        "ocr_axis_path": str(out_base.with_name(out_base.name + "_ocr_axis.json")),
        "mllm_cache_root": str(enhanced_mllm_cache_root) if enhanced_mllm_cache_root is not None else "",
    }


def _enhanced_reconstruction_artifacts_complete(outputs, require_priority=True):
    if not isinstance(outputs, dict):
        return False
    required_keys = [
        "grid_label_bindings_path",
        "grid_layers_path",
        "mllm_axis_path",
        "ocr_axis_path",
    ]
    if require_priority:
        required_keys.append("final_selection_path")
    if not any(outputs.get(key) and Path(outputs[key]).exists() for key in ("final_overlay_path", "final_grid_path")):
        return False
    for key in required_keys:
        path = outputs.get(key)
        if not path or not Path(path).exists():
            return False
    return True


def _run_enhanced_cartesian_grid_reconstruction(image_path, output_dir, disable_cache=False):
    if not _env_flag_enabled("CARTESIAN_ENHANCED_GRID", True):
        return {"enabled": False, "reason": "disabled_by_CARTESIAN_ENHANCED_GRID"}

    image_path_obj = Path(image_path).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    api_key = get_api_key()
    if api_key and not os.environ.get("MLLM_API_KEY"):
        os.environ["MLLM_API_KEY"] = api_key
    if not os.environ.get("MLLM_MODEL"):
        os.environ["MLLM_MODEL"] = get_model_name()
    if not os.environ.get("MLLM_ENDPOINT"):
        os.environ["MLLM_ENDPOINT"] = get_chat_completion_url()
    mllm_model_name = os.getenv("MLLM_MODEL") or get_model_name()
    enhanced_mllm_cache_root = None
    if not disable_cache:
        enhanced_mllm_cache_root = Path(
            os.getenv(
                "GRID_RECONSTRUCT_MLLM_CACHE_ROOT",
                str(Path(LLM_CACHE_DIR) / "enhanced_grid_reconstruction"),
            )
        ) / _safe_cache_component(mllm_model_name)

    existing_outputs = _enhanced_reconstruction_outputs(
        output_root,
        image_path_obj.with_suffix("").name,
        enhanced_mllm_cache_root,
    )
    if not disable_cache and _env_flag_enabled("GRID_RECONSTRUCT_REUSE_ARTIFACTS", True) and _enhanced_reconstruction_artifacts_complete(
        existing_outputs,
        require_priority=_env_flag_enabled("GRID_RECONSTRUCT_GRID_ARBITRATION", True),
    ):
        logger.debug("Reusing enhanced grid reconstruction artifacts for %s", image_path_obj)
        return {"enabled": True, "error": None, "source": "artifact_cache", "outputs": existing_outputs}

    try:
        from grid_line_filter import process_image as process_reconstruction_image
    except Exception as exc:
        return {"enabled": True, "error": f"enhanced_grid_import_failed: {exc}"}

    args = argparse.Namespace(
        sat_max=int(os.getenv("GRID_RECONSTRUCT_SAT_MAX", "70")),
        white_cutoff=int(os.getenv("GRID_RECONSTRUCT_WHITE_CUTOFF", "255")),
        min_gray=int(os.getenv("GRID_RECONSTRUCT_MIN_GRAY", "95")),
        contrast_min=int(os.getenv("GRID_RECONSTRUCT_CONTRAST_MIN", "7")),
        include_dark=_env_flag_enabled("GRID_RECONSTRUCT_INCLUDE_DARK", False),
        dark_cutoff=int(os.getenv("GRID_RECONSTRUCT_DARK_CUTOFF", "80")),
        min_line_frac=float(os.getenv("GRID_RECONSTRUCT_MIN_LINE_FRAC", "0.055")),
        gap_frac=float(os.getenv("GRID_RECONSTRUCT_GAP_FRAC", "0.006")),
        max_thickness_frac=float(os.getenv("GRID_RECONSTRUCT_MAX_THICKNESS_FRAC", "0.008")),
        min_grid_span_frac=float(os.getenv("GRID_RECONSTRUCT_MIN_GRID_SPAN_FRAC", "0.18")),
        min_grid_lines=int(os.getenv("GRID_RECONSTRUCT_MIN_GRID_LINES", "2")),
        cluster_tolerance=int(os.getenv("GRID_RECONSTRUCT_CLUSTER_TOLERANCE", "3")),
        grid_thickness=int(os.getenv("GRID_RECONSTRUCT_GRID_THICKNESS", "1")),
        tick_dark_cutoff=int(os.getenv("GRID_RECONSTRUCT_TICK_DARK_CUTOFF", "150")),
        no_ocr=not _env_flag_enabled("GRID_RECONSTRUCT_OCR", True),
        ocr_lang=os.getenv("GRID_RECONSTRUCT_OCR_LANG", "en"),
        ocr_min_score=float(os.getenv("GRID_RECONSTRUCT_OCR_MIN_SCORE", "0.45")),
        ocr_det_thresh=float(os.getenv("GRID_RECONSTRUCT_OCR_DET_THRESH", "0.35")),
        ocr_det_box_thresh=float(os.getenv("GRID_RECONSTRUCT_OCR_DET_BOX_THRESH", "0.60")),
        ocr_det_unclip_ratio=float(os.getenv("GRID_RECONSTRUCT_OCR_DET_UNCLIP_RATIO", "1.15")),
        ocr_det_limit_side_len=int(os.getenv("GRID_RECONSTRUCT_OCR_DET_LIMIT_SIDE_LEN", "960")),
        ocr_det_limit_type=os.getenv("GRID_RECONSTRUCT_OCR_DET_LIMIT_TYPE", "max"),
        ocr_return_word_box=_env_flag_enabled("GRID_RECONSTRUCT_OCR_RETURN_WORD_BOX", False),
        mllm=_env_flag_enabled("GRID_RECONSTRUCT_MLLM", True),
        mllm_model=mllm_model_name,
        mllm_endpoint=os.getenv("MLLM_ENDPOINT") or get_chat_completion_url(),
        mllm_api_key_env="MLLM_API_KEY",
        mllm_cache_root=enhanced_mllm_cache_root,
        mllm_timeout=float(os.getenv("MLLM_TIMEOUT_SECONDS", "180")),
        no_semantic_guard=not _env_flag_enabled("GRID_RECONSTRUCT_SEMANTIC_GUARD", True),
        no_grid_arbitration=not _env_flag_enabled("GRID_RECONSTRUCT_GRID_ARBITRATION", True),
        panel_width=int(os.getenv("GRID_RECONSTRUCT_PANEL_WIDTH", "360")),
    )

    try:
        preview_path = process_reconstruction_image(
            image_path_obj,
            image_path_obj.parent,
            output_root,
            args,
        )
    except Exception as exc:
        logger.exception("Enhanced cartesian grid reconstruction failed")
        return {"enabled": True, "error": str(exc)}

    key_outputs = _enhanced_reconstruction_outputs(
        output_root,
        image_path_obj.with_suffix("").name,
        enhanced_mllm_cache_root,
        preview_path=preview_path,
    )
    return {"enabled": True, "error": None, "source": "computed", "outputs": key_outputs}


def _enhanced_basic_grid_visual_path(enhanced_grid_reconstruction):
    if not isinstance(enhanced_grid_reconstruction, dict):
        return None
    if enhanced_grid_reconstruction.get("error"):
        return None
    outputs = enhanced_grid_reconstruction.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for key in ("final_grid_path", "final_overlay_path"):
        path = outputs.get(key)
        if path and Path(path).exists():
            return Path(path)
    return None


def _safe_cache_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("._") or "unknown"

# 旧版 tick 读取辅助函数：当前主流程主要使用
# function_calling.label.extract_tick_labels_with_llm.extract_tick_labels_with_llm。
# 这里保留是为了兼容早期调试入口，不作为新逻辑的优先修改点。
async def call_llm_recognize_ticks(image_path, direction, tick_regions):
    """
    使用大模型识别图像中的刻度值
    """
    # 图像读取与编码
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    # 构造提示词
    direction_text = "X轴" if direction == 'x' else "Y轴"
    prompt = f"请识别图片中{direction_text}上的所有刻度值。请仔细查看{direction_text}上的数字标签，只返回所有可见的数字，每个数字占一行。请严格按照这个格式返回：\n```\n数字1\n数字2\n数字3\n...\n```\n不要包含任何其他文字说明。"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # 发送请求
    try:
        content = await chat_with_gemini(
            messages,
            model=model_name,
            temperature=0,
        )
        if content == FAILURE_TEXT:
            raise ValueError("Model API request failed.")
        content = content.strip()
                
        # 提取数字
        import re
        numbers = []
        # 尝试从代码块中提取
        code_blocks = re.findall(r"```(?:\w+)?\s*([\s\S]*?)```", content)
        if code_blocks:
            text_content = code_blocks[0]
        else:
            text_content = content
                
        # 提取所有数字
        for line in text_content.splitlines():
            line = line.strip()
            if line:
                try:
                    # 尝试直接转换为数字
                    num = float(line)
                    numbers.append(num)
                except ValueError:
                    # 尝试提取行中的数字
                    nums = re.findall(r"-?\d+\.?\d*", line)
                    for n in nums:
                        try:
                            numbers.append(float(n))
                        except ValueError:
                            continue
                
        logger.debug(f"大模型识别到{direction_text}刻度值: {numbers}")
        return numbers
    except Exception as e:
        logger.error(f"大模型识别失败: {e}")
        return []

def recognize_tick_labels_with_llm(img, ticks, direction, temp_dir=None):
    """
    使用大模型替代OCR识别刻度标签
    """
    # 创建临时文件来保存图像
    import tempfile
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, img)
        
        # 运行异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tick_values = loop.run_until_complete(call_llm_recognize_ticks(temp_path, direction, ticks))
        finally:
            loop.close()
        
        # 将识别的数值与刻度位置匹配
        result = []
        if direction == 'x' and tick_values:
            # X轴刻度按位置排序
            sorted_ticks = sorted(ticks, key=lambda t: (t[0] + t[2]) // 2)
            # 确保数值数量与刻度数量匹配
            if len(tick_values) > len(sorted_ticks):
                # 如果数值过多，选择最接近数量的
                tick_values = tick_values[:len(sorted_ticks)]
            elif len(tick_values) < len(sorted_ticks):
                # 如果数值不足，使用已有数值插值
                import numpy as np
                if len(tick_values) >= 2:
                    # 插值生成更多数值
                    x_positions = np.linspace(0, 1, len(sorted_ticks))
                    orig_positions = np.linspace(0, 1, len(tick_values))
                    interpolated = np.interp(x_positions, orig_positions, tick_values)
                    tick_values = interpolated.tolist()
            
            # 创建结果列表
            for i, tick in enumerate(sorted_ticks):
                if i < len(tick_values):
                    result.append({
                        'tick': tick,
                        'text': str(tick_values[i])
                    })
        
        elif direction == 'y' and tick_values:
            # Y轴刻度按位置排序（从下到上）
            sorted_ticks = sorted(ticks, key=lambda t: (t[1] + t[3]) // 2, reverse=True)
            # 确保数值数量与刻度数量匹配
            if len(tick_values) > len(sorted_ticks):
                tick_values = tick_values[:len(sorted_ticks)]
            elif len(tick_values) < len(sorted_ticks):
                # 如果数值不足，使用已有数值插值
                import numpy as np
                if len(tick_values) >= 2:
                    # 插值生成更多数值
                    y_positions = np.linspace(0, 1, len(sorted_ticks))
                    orig_positions = np.linspace(0, 1, len(tick_values))
                    interpolated = np.interp(y_positions, orig_positions, tick_values)
                    tick_values = interpolated.tolist()
            
            # 创建结果列表
            for i, tick in enumerate(sorted_ticks):
                if i < len(tick_values):
                    result.append({
                        'tick': tick,
                        'text': str(tick_values[i])
                    })
        
        return result
    except Exception as e:
        logger.error(f"使用大模型识别刻度标签时出错: {e}")
        return []
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

# 导入所需的功能模块
# ===== 主流程依赖的原子模块 =====
# axis: 线段检测、线段合并、坐标轴初判。
# ticks: 沿轴扫描短 tick，并做基础过滤。
# label: 调用 MLLM 读取 tick 文本和柱端数值标签。
# color: 读取图例颜色、点图对象和系列信息。
# image/utils: 早期绘图和图像读写工具。
from function_calling.axis.detect_lines import detect_candidate_lines
from function_calling.axis.merge_lines import merge_similar_lines
from function_calling.axis.infer_axes import infer_axes_from_lines
from function_calling.ticks.detect_ticks import scan_pixels_for_ticks
from function_calling.ticks.filter_ticks import filter_ticks
from function_calling.label.recognize_tick_labels import recognize_tick_labels
from function_calling.label.extract_tick_labels_with_llm import (
    extract_bar_value_labels_with_llm,
    extract_tick_labels_with_llm,
)
from function_calling.color.extract_chart_colors import extract_chart_series_color, extract_point_chart_items
from function_calling.image.draw_grid_from_ticks import draw_grid_from_ticks
from utils.image_io import load_image, save_image

try:
    from grid_visual import binding_source_color as _grid_binding_source_color
    from grid_visual import draw_label_box as _grid_draw_label_box
    from grid_visual import draw_text_like_ocr_box as _grid_draw_text_like_ocr_box
    from grid_visual import ocr_box_float_points as _grid_ocr_box_float_points
    from grid_visual import sampled_text_color as _grid_sampled_text_color
except Exception:
    _grid_binding_source_color = None
    _grid_draw_label_box = None
    _grid_draw_text_like_ocr_box = None
    _grid_ocr_box_float_points = None
    _grid_sampled_text_color = None


NUMERIC_AXIS_TYPE = "\u6570\u503c\u8f74"
TEXT_AXIS_TYPE = "\u6587\u5b57\u8f74"


def _enhanced_binding_color(source="mllm"):
    if _grid_binding_source_color is not None:
        return _grid_binding_source_color(source)
    return (165, 45, 185)


def _draw_enhanced_style_label(image, text, origin, color, anchor="center", font_scale=0.46):
    if _grid_draw_label_box is not None:
        _grid_draw_label_box(
            image,
            text,
            origin,
            color,
            anchor=anchor,
            font_scale=font_scale,
            draw_border=False,
            text_color=(0, 0, 0),
        )
        return

    if not text:
        return
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(origin[0]), int(origin[1])
    if anchor == "center":
        x -= text_w // 2
    elif anchor == "right":
        x -= text_w
    x = max(2, min(w - text_w - 6, x))
    y = max(text_h + 6, min(h - baseline - 4, y))
    pad_x = 4
    pad_y = 3
    top_left = (x - pad_x, y - text_h - pad_y)
    bottom_right = (x + text_w + pad_x, y + baseline + pad_y)
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_enhanced_grid_bindings(enhanced_grid_reconstruction):
    if not isinstance(enhanced_grid_reconstruction, dict) or enhanced_grid_reconstruction.get("error"):
        return None
    outputs = enhanced_grid_reconstruction.get("outputs")
    if not isinstance(outputs, dict):
        return None

    def usable_binding_count(bindings):
        if not isinstance(bindings, dict):
            return 0
        total = 0
        for axis_key in ("x_axis", "y_axis"):
            axis_binding = bindings.get(axis_key)
            if isinstance(axis_binding, dict) and isinstance(axis_binding.get("tick_bindings"), list):
                total += len(axis_binding.get("tick_bindings"))
        return total

    def load_binding_file(path_obj):
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                bindings = json.load(f)
            return bindings if isinstance(bindings, dict) else None
        except Exception as exc:
            logger.warning("Failed to load enhanced grid bindings from %s: %s", path_obj, exc)
            return None

    exact_paths = []
    fallback_paths = []
    for key in ("grid_label_bindings_path",):
        path = outputs.get(key)
        if not path:
            continue
        path_obj = Path(path)
        exact_paths.append(path_obj)
        if path_obj.parent.exists():
            base_name = path_obj.name.replace("_grid_label_bindings.json", "")
            base_tokens = [token for token in base_name.split("_") if token]
            for candidate in sorted(path_obj.parent.glob("*_grid_label_bindings.json")):
                if candidate == path_obj:
                    continue
                candidate_name = candidate.name
                if base_name and (base_name in candidate_name or candidate.stem in base_name):
                    fallback_paths.append(candidate)
                    continue
                if len(base_tokens) >= 2 and all(token in candidate_name for token in base_tokens[:2]):
                    fallback_paths.append(candidate)

    seen = set()
    for path_obj in exact_paths:
        path_obj = Path(path_obj)
        path_key = str(path_obj.resolve()) if path_obj.exists() else str(path_obj)
        if path_key in seen or not path_obj.exists():
            continue
        seen.add(path_key)
        bindings = load_binding_file(path_obj)
        if usable_binding_count(bindings) > 0:
            return bindings

    best_bindings = None
    best_count = 0
    for path_obj in fallback_paths:
        path_obj = Path(path_obj)
        path_key = str(path_obj.resolve()) if path_obj.exists() else str(path_obj)
        if path_key in seen or not path_obj.exists():
            continue
        seen.add(path_key)
        bindings = load_binding_file(path_obj)
        if isinstance(bindings, dict):
            binding_count = usable_binding_count(bindings)
            if binding_count > best_count:
                best_bindings = bindings
                best_count = binding_count
            elif best_bindings is None and {"x_axis", "y_axis"} & set(bindings.keys()):
                best_bindings = bindings
    return best_bindings


def _midpoint_value(left_value, right_value, axis_scale="linear"):
    left = float(left_value)
    right = float(right_value)
    if axis_scale == "log" and left > 0 and right > 0:
        value = math.sqrt(left * right)
        if max(abs(left), abs(right)) >= 100:
            value = round(value)
        elif max(abs(left), abs(right)) >= 1:
            value = round(value, 4)
    else:
        value = (left + right) / 2
    return round(float(f"{value:.12f}"), 10)


def _clone_ocr_tick(ocr_tick):
    if not isinstance(ocr_tick, dict):
        return None
    try:
        return json.loads(json.dumps(ocr_tick, ensure_ascii=False))
    except Exception:
        return dict(ocr_tick)


def _ocr_points_for_encryption(ocr_tick):
    if not isinstance(ocr_tick, dict):
        return None
    if _grid_ocr_box_float_points is not None:
        points = _grid_ocr_box_float_points(ocr_tick)
        if points is not None:
            return points.astype(np.float32)
    box = ocr_tick.get("box")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        points = np.array([[float(point[0]), float(point[1])] for point in box], dtype=np.float32)
    except (TypeError, ValueError, IndexError):
        return None
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2:
        return None
    return points


def _style_axis_key(axis_binding):
    axis_name = str(axis_binding.get("axis", "") if isinstance(axis_binding, dict) else "").lower()
    if axis_name.startswith("y"):
        return "y"
    return "x"


def _shift_ocr_points_to_axis_position(points, source_position, target_position, axis_key):
    shifted = points.astype(np.float32).copy()
    delta = float(target_position) - float(source_position)
    if axis_key == "y":
        shifted[:, 1] += delta
    else:
        shifted[:, 0] += delta
    return shifted


def _ocr_box_geometry_for_encryption(points):
    if points is None or getattr(points, "shape", (0,))[0] < 4:
        return None
    edges = []
    for index in range(points.shape[0]):
        delta = points[(index + 1) % points.shape[0]] - points[index]
        length = float(np.linalg.norm(delta))
        if length > 0:
            edges.append((length, delta))
    if not edges:
        return None
    long_length, long_delta = max(edges, key=lambda item: item[0])
    short_lengths = [length for length, _ in edges if length < long_length * 0.75]
    short_length = float(np.median(short_lengths)) if short_lengths else max(6.0, long_length * 0.35)
    angle = float(np.degrees(np.arctan2(float(long_delta[1]), float(long_delta[0]))))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    center = np.array([float(points[:, 0].mean()), float(points[:, 1].mean())], dtype=np.float32)
    return center, angle, max(8.0, float(long_length)), max(6.0, float(short_length))


def _snap_label_angle(angle):
    try:
        value = float(angle)
    except (TypeError, ValueError):
        return 0.0
    while value > 90:
        value -= 180
    while value < -90:
        value += 180
    if abs(value) < 35:
        return 0.0
    if abs(abs(value) - 90) < 35:
        return 90.0 if value >= 0 else -90.0
    return 0.0


def _ocr_tick_text(ocr_tick):
    if not isinstance(ocr_tick, dict):
        return ""
    return str(ocr_tick.get("text", "") or "").strip()


def _binding_label_text(item):
    if not isinstance(item, dict):
        return ""
    label = str(item.get("label", "") or "").strip()
    if label:
        return label
    return _ocr_tick_text(item.get("ocr"))


def _looks_like_numeric_label(text):
    value = str(text or "").strip().replace(",", "")
    if not value:
        return False
    return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?", value))


def _axis_stable_label_angle(bindings, *, numeric_axis=False):
    if numeric_axis:
        return 0.0
    angles = []
    if not isinstance(bindings, list):
        return 0.0
    for item in bindings:
        if not isinstance(item, dict):
            continue
        text = _binding_label_text(item)
        if _looks_like_numeric_label(text):
            angles.append(0.0)
            continue
        points = _ocr_points_for_encryption(item.get("ocr"))
        geometry = _ocr_box_geometry_for_encryption(points)
        if geometry is None:
            continue
        _center, raw_angle, width, height = geometry
        if max(width, height) / max(1.0, min(width, height)) < 1.25:
            snapped = 0.0
        else:
            snapped = _snap_label_angle(raw_angle)
        angles.append(snapped)
    if not angles:
        return 0.0
    vertical = [angle for angle in angles if abs(angle) == 90.0]
    if len(vertical) > len(angles) / 2:
        return 90.0 if sum(1 for angle in vertical if angle > 0) >= len(vertical) / 2 else -90.0
    return 0.0


def _update_ocr_tick_points(ocr_tick, points, *, text_angle=None, synthetic=False):
    if points is None:
        return _clone_ocr_tick(ocr_tick)
    style = _clone_ocr_tick(ocr_tick) or {}
    style["box"] = [[float(point[0]), float(point[1])] for point in points[:4]]
    center = points.mean(axis=0)
    style["center"] = [float(center[0]), float(center[1])]
    style["x"] = float(center[0])
    style["y"] = float(center[1])
    if text_angle is not None:
        style["text_angle"] = float(text_angle)
    if synthetic:
        style["synthetic_encrypted_label_box"] = True
        style["mllm_pseudo_box"] = False
    return style


def _rect_points_from_center(center, width, height, angle):
    cx, cy = float(center[0]), float(center[1])
    width = max(8.0, float(width))
    height = max(6.0, float(height))
    theta = math.radians(float(angle))
    ux = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    uy = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float32)
    center_vec = np.array([cx, cy], dtype=np.float32)
    half_w = width / 2.0
    half_h = height / 2.0
    return np.array(
        [
            center_vec - ux * half_w - uy * half_h,
            center_vec + ux * half_w - uy * half_h,
            center_vec + ux * half_w + uy * half_h,
            center_vec - ux * half_w + uy * half_h,
        ],
        dtype=np.float32,
    )


def _stabilize_label_points(points, stable_angle):
    geometry = _ocr_box_geometry_for_encryption(points)
    if geometry is None:
        return points
    center, _angle, width, height = geometry
    if abs(float(stable_angle)) < 1e-6 and width < height:
        width, height = height, width
    elif abs(abs(float(stable_angle)) - 90.0) < 1e-6 and height < width:
        width, height = height, width
    return _rect_points_from_center(center, width, height, stable_angle)


def _normalize_label_style_for_axis(ocr_tick, stable_angle):
    points = _ocr_points_for_encryption(ocr_tick)
    if points is None:
        style = _clone_ocr_tick(ocr_tick)
        if isinstance(style, dict):
            style["text_angle"] = float(stable_angle)
        return style
    points = _stabilize_label_points(points, stable_angle)
    return _update_ocr_tick_points(ocr_tick, points, text_angle=stable_angle, synthetic=False)


def _ocr_tick_from_points(template_ocr, points, color_source_ocr=None, text_angle=None):
    if points is None:
        return None
    synthetic = _update_ocr_tick_points(
        template_ocr,
        points,
        text_angle=text_angle,
        synthetic=True,
    )
    if color_source_ocr is not None:
        synthetic["color_source_ocr"] = _clone_ocr_tick(color_source_ocr)
    return synthetic


def _interpolate_encrypted_label_ocr(left_item, right_item, target_position, axis_key, stable_angle=0.0):
    left_ocr = left_item.get("ocr") if isinstance(left_item, dict) else None
    right_ocr = right_item.get("ocr") if isinstance(right_item, dict) else None
    left_points = _ocr_points_for_encryption(left_ocr)
    right_points = _ocr_points_for_encryption(right_ocr)
    left_position = left_item.get("position")
    right_position = right_item.get("position")
    try:
        denominator = float(right_position) - float(left_position)
    except (TypeError, ValueError):
        denominator = 0.0

    if left_points is not None and right_points is not None and abs(denominator) > 1e-6:
        ratio = (float(target_position) - float(left_position)) / denominator
        ratio = max(0.0, min(1.0, ratio))
        points = left_points * (1.0 - ratio) + right_points * ratio
        points = _stabilize_label_points(points, stable_angle)
        color_source = left_ocr if ratio <= 0.5 else right_ocr
        return _ocr_tick_from_points(color_source, points, color_source, stable_angle)

    if left_points is not None:
        points = _shift_ocr_points_to_axis_position(left_points, left_position, target_position, axis_key)
        points = _stabilize_label_points(points, stable_angle)
        return _ocr_tick_from_points(left_ocr, points, left_ocr, stable_angle)

    if right_points is not None:
        points = _shift_ocr_points_to_axis_position(right_points, right_position, target_position, axis_key)
        points = _stabilize_label_points(points, stable_angle)
        return _ocr_tick_from_points(right_ocr, points, right_ocr, stable_angle)

    return None


def _regularize_axis_label_ocr_positions(bindings, axis_key, stable_angle=0.0):
    if not isinstance(bindings, list) or len(bindings) < 3:
        return bindings

    geometries = []
    for index, item in enumerate(bindings):
        if not isinstance(item, dict):
            continue
        points = _ocr_points_for_encryption(item.get("ocr"))
        geometry = _ocr_box_geometry_for_encryption(points)
        if geometry is None:
            continue
        center, _angle, width, height = geometry
        axis_position = _safe_float(item.get("position"))
        if axis_position is None:
            continue
        perp = float(center[0]) if axis_key == "y" else float(center[1])
        axis_coord = float(center[1]) if axis_key == "y" else float(center[0])
        geometries.append(
            {
                "index": index,
                "axis_position": axis_position,
                "axis_coord": axis_coord,
                "perp": perp,
                "width": float(width),
                "height": float(height),
            }
        )

    if len(geometries) < 3:
        return bindings

    perps = np.array([item["perp"] for item in geometries], dtype=np.float32)
    median_perp = float(np.median(perps))
    deviations = np.abs(perps - median_perp)
    median_deviation = float(np.median(deviations))
    axis_positions = np.array([item["axis_position"] for item in geometries], dtype=np.float32)
    axis_step = 0.0
    if len(axis_positions) >= 2:
        diffs = np.diff(np.sort(axis_positions))
        diffs = diffs[diffs > 1e-3]
        if diffs.size:
            axis_step = float(np.median(diffs))
    outlier_threshold = max(24.0, median_deviation * 6.0, axis_step * 0.18)
    inlier_geometries = [item for item, deviation in zip(geometries, deviations) if float(deviation) <= outlier_threshold]
    if len(inlier_geometries) < 2:
        inlier_geometries = geometries

    median_width = float(np.median(np.array([item["width"] for item in inlier_geometries], dtype=np.float32)))
    median_height = float(np.median(np.array([item["height"] for item in inlier_geometries], dtype=np.float32)))
    regularized = list(bindings)
    for geometry, deviation in zip(geometries, deviations):
        if float(deviation) <= outlier_threshold:
            continue
        item = regularized[geometry["index"]]
        axis_position = geometry["axis_position"]
        center = [median_perp, axis_position] if axis_key == "y" else [axis_position, median_perp]
        points = _rect_points_from_center(center, median_width, median_height, stable_angle)
        fixed_ocr = _ocr_tick_from_points(item.get("ocr"), points, item.get("ocr"), stable_angle)
        if isinstance(fixed_ocr, dict):
            fixed_ocr["axis_label_position_regularized"] = True
            fixed_ocr["axis_label_position_regularize_reason"] = "perpendicular_axis_label_outlier"
            fixed_ocr["axis_label_position_previous_center"] = [
                geometry["axis_coord"] if axis_key == "x" else geometry["perp"],
                geometry["perp"] if axis_key == "x" else geometry["axis_coord"],
            ]
            fixed_ocr["axis_label_position_regularize_median_perp"] = median_perp
            fixed_ocr["axis_label_position_regularize_threshold"] = outlier_threshold
        updated = dict(item)
        updated["ocr"] = fixed_ocr
        updated["label_position_regularized"] = True
        regularized[geometry["index"]] = updated
    return regularized


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _ocr_label_axis_extent(ocr_tick, axis_key):
    points = _ocr_points_for_encryption(ocr_tick)
    if points is None:
        return None
    if axis_key == "y":
        extent = float(points[:, 1].max() - points[:, 1].min())
    else:
        extent = float(points[:, 0].max() - points[:, 0].min())
    return extent if extent > 0 else None


def _encrypted_midpoint_spacing_ok(left_item, right_item, axis_key):
    spacing_ratio = _env_float("CARTESIAN_ENCRYPT_MIN_LABEL_GAP_RATIO", 1.2)
    min_padding = _env_float("CARTESIAN_ENCRYPT_MIN_LABEL_GAP_PADDING", 2.0)
    tolerance = _env_float("CARTESIAN_ENCRYPT_MIN_LABEL_GAP_TOLERANCE", 2.0)
    left_position = _safe_float(left_item.get("position") if isinstance(left_item, dict) else None)
    right_position = _safe_float(right_item.get("position") if isinstance(right_item, dict) else None)
    if left_position is None or right_position is None:
        return True, {"reason": "missing_position"}

    encrypted_neighbor_spacing = abs(right_position - left_position) / 2.0
    extents = [
        extent
        for extent in (
            _ocr_label_axis_extent(left_item.get("ocr"), axis_key),
            _ocr_label_axis_extent(right_item.get("ocr"), axis_key),
        )
        if extent is not None
    ]
    if not extents:
        return True, {
            "reason": "missing_ocr_label_extent",
            "encrypted_neighbor_spacing": encrypted_neighbor_spacing,
        }

    label_extent = max(extents)
    required_spacing = label_extent * spacing_ratio + min_padding
    spacing_ok = encrypted_neighbor_spacing + tolerance >= required_spacing
    return spacing_ok, {
        "reason": "ok" if spacing_ok else "dense_label_spacing",
        "encrypted_neighbor_spacing": round(encrypted_neighbor_spacing, 3),
        "label_axis_extent": round(label_extent, 3),
        "required_spacing": round(required_spacing, 3),
        "spacing_ratio": spacing_ratio,
        "spacing_tolerance": tolerance,
        "axis_key": axis_key,
    }


def _prepare_ocr_style_for_draw(ocr_tick, source_image):
    style = _clone_ocr_tick(ocr_tick)
    if not isinstance(style, dict):
        return None
    if style.get("synthetic_encrypted_label_box"):
        style["text_color"] = [0, 0, 0]
        return style
    explicit_color = style.get("text_color")
    if isinstance(explicit_color, (list, tuple)) and len(explicit_color) >= 3:
        return style
    if _grid_sampled_text_color is None:
        return style
    color_source = style.get("color_source_ocr")
    if not isinstance(color_source, dict):
        color_source = style
    points = _ocr_points_for_encryption(color_source)
    if points is None:
        return style
    try:
        color = _grid_sampled_text_color(source_image, points)
        style["text_color"] = [int(value) for value in color]
    except Exception:
        pass
    return style


def _draw_ocr_style_label(image, source_image, text, ocr_tick, fallback_color):
    style = _prepare_ocr_style_for_draw(ocr_tick, source_image)
    if style is None:
        return False
    style["text_color"] = [0, 0, 0]
    points = _ocr_points_for_encryption(style)
    geometry = _ocr_box_geometry_for_encryption(points)
    if geometry is not None:
        center, angle, box_width, box_height = geometry
        if abs(float(angle)) < 10.0:
            h, w = image.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            (base_w, base_h), _baseline = cv2.getTextSize(text, font, 1.0, thickness)
            if base_w > 0 and base_h > 0:
                scale = min(float(box_width) * 0.92 / base_w, float(box_height) * 0.78 / max(1, base_h))
                scale = max(0.28, min(0.72, float(scale)))
                (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
                x = int(round(float(center[0]) - text_w / 2.0))
                y = int(round(float(center[1]) + text_h / 2.0))
                x = max(2, min(w - text_w - 2, x))
                y = max(text_h + 2, min(h - baseline - 2, y))
                pad_x = 2
                pad_y = 1
                top_left = (max(0, x - pad_x), max(0, y - text_h - pad_y))
                bottom_right = (min(w - 1, x + text_w + pad_x), min(h - 1, y + baseline + pad_y))
                cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(image, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
                return True

    if _grid_draw_text_like_ocr_box is None:
        return False
    try:
        return bool(
            _grid_draw_text_like_ocr_box(
                image,
                source_image,
                text,
                style,
                fallback_color=fallback_color,
                draw_box=False,
                text_color=(0, 0, 0),
                alpha_scale=1.0,
                fill_background=True,
            )
        )
    except Exception:
        return False


def _axis_encryption_from_enhanced_bindings(axis_binding, axis_scale="linear"):
    if not isinstance(axis_binding, dict):
        return None
    raw_bindings = axis_binding.get("tick_bindings")
    if not isinstance(raw_bindings, list):
        return None

    bindings = []
    for binding in raw_bindings:
        if not isinstance(binding, dict) or binding.get("display_suppressed"):
            continue
        position = _safe_float(binding.get("position"))
        if position is None:
            continue
        label = str(binding.get("label", "") or "").strip()
        numeric = _safe_float(binding.get("numeric"))
        if not label and numeric is None:
            continue
        bindings.append(
            {
                "position": position,
                "label": label if label else numeric,
                "numeric": numeric,
                "source": str(binding.get("source", "none") or "none"),
                "ocr": _clone_ocr_tick(binding.get("ocr")),
            }
        )

    if not bindings:
        return None
    bindings.sort(key=lambda item: item["position"])

    bounds = None
    raw_bounds = axis_binding.get("bounds")
    if isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) >= 2:
        first = _safe_float(raw_bounds[0])
        second = _safe_float(raw_bounds[1])
        if first is not None and second is not None:
            bounds = (int(round(min(first, second))), int(round(max(first, second))))

    is_numeric = len(bindings) >= 2 and all(item["numeric"] is not None for item in bindings)
    native_ticks = [item["numeric"] if is_numeric else item["label"] for item in bindings]
    native_pixels = [int(round(item["position"])) for item in bindings]

    encrypted_ticks = []
    encrypted_pixels = []
    encrypted_label_styles = []
    skipped_encrypted_intervals = []
    axis_key = _style_axis_key(axis_binding)
    stable_label_angle = _axis_stable_label_angle(bindings, numeric_axis=is_numeric)
    bindings = _regularize_axis_label_ocr_positions(bindings, axis_key, stable_label_angle)
    native_label_styles = [_normalize_label_style_for_axis(item.get("ocr"), stable_label_angle) for item in bindings]
    if is_numeric:
        dense_interval_warnings = []
        for index, item in enumerate(bindings[:-1]):
            next_item = bindings[index + 1]
            spacing_ok, spacing_details = _encrypted_midpoint_spacing_ok(item, next_item, axis_key)
            if not spacing_ok:
                dense_interval_warnings.append(
                    {
                        "left_tick": item["numeric"],
                        "right_tick": next_item["numeric"],
                        "left_pixel": int(round(item["position"])),
                        "right_pixel": int(round(next_item["position"])),
                        **spacing_details,
                    }
                )
        for index, item in enumerate(bindings):
            encrypted_ticks.append(item["numeric"])
            encrypted_pixels.append(int(round(item["position"])))
            encrypted_label_styles.append(_normalize_label_style_for_axis(item.get("ocr"), stable_label_angle))
            if index < len(bindings) - 1:
                next_item = bindings[index + 1]
                mid_position = (item["position"] + next_item["position"]) / 2
                encrypted_ticks.append(_midpoint_value(item["numeric"], next_item["numeric"], axis_scale))
                encrypted_pixels.append(int(round(mid_position)))
                encrypted_label_styles.append(
                    _interpolate_encrypted_label_ocr(item, next_item, mid_position, axis_key, stable_label_angle)
                )
        skipped_encrypted_intervals = []
    else:
        encrypted_ticks = native_ticks.copy()
        encrypted_pixels = native_pixels.copy()
        encrypted_label_styles = native_label_styles.copy()
        dense_interval_warnings = []

    return {
        "axis_type": NUMERIC_AXIS_TYPE if is_numeric else TEXT_AXIS_TYPE,
        "native_ticks": native_ticks,
        "native_pixels": native_pixels,
        "native_label_styles": native_label_styles,
        "encrypted_ticks": encrypted_ticks,
        "encrypted_pixels": encrypted_pixels,
        "encrypted_label_styles": encrypted_label_styles,
        "skipped_encrypted_intervals": skipped_encrypted_intervals,
        "dense_interval_warnings": dense_interval_warnings,
        "axis_encryption_policy": "all_numeric_midpoints" if is_numeric else "text_axis_no_encryption",
        "stable_label_angle": stable_label_angle,
        "bounds": bounds,
        "binding_count": len(bindings),
    }


def _encrypted_ticks_from_enhanced_bindings(enhanced_grid_reconstruction, x_axis_scale="linear", y_axis_scale="linear"):
    bindings = _load_enhanced_grid_bindings(enhanced_grid_reconstruction)
    if not bindings:
        return None

    result = {
        "source": "enhanced_grid_label_bindings",
        "x": _axis_encryption_from_enhanced_bindings(bindings.get("x_axis"), x_axis_scale),
        "y": _axis_encryption_from_enhanced_bindings(bindings.get("y_axis"), y_axis_scale),
    }
    if not result["x"] and not result["y"]:
        return None
    return result


# ===== 上传阶段轴结构先验的规范化 =====
# 上传时的 MLLM 会返回 axis_repair_hint，用来描述是否缺轴、缺 tick、
# 是否有背景网格、轴在左/右/中间、轴角色是数值还是类别等。
# 后续所有补轴、补 tick、背景网格恢复逻辑都先走这里统一字段格式，
# 这样可以保证正常图表默认不触发修复逻辑。
def _axis_line_from_enhanced_ticks(axis_key, axis_info, cross_axis_info, image_shape):
    h, w = image_shape[:2]
    info = axis_info if isinstance(axis_info, dict) else {}
    cross = cross_axis_info if isinstance(cross_axis_info, dict) else {}
    pixels = info.get("encrypted_pixels") or info.get("native_pixels") or []
    cross_pixels = cross.get("encrypted_pixels") or cross.get("native_pixels") or []
    bounds = info.get("bounds")
    cross_bounds = cross.get("bounds")

    if axis_key == "x":
        x0 = int(min(pixels)) if pixels else 0
        x1 = int(max(pixels)) if pixels else max(0, w - 1)
        if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            y = int(max(bounds[0], bounds[1]))
        elif cross_pixels:
            y = int(max(cross_pixels))
        else:
            y = max(0, h - 1)
        return [x0, y, x1, y]

    y0 = int(min(pixels)) if pixels else 0
    y1 = int(max(pixels)) if pixels else max(0, h - 1)
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
        x = int(min(bounds[0], bounds[1]))
    elif isinstance(cross_bounds, (list, tuple)) and len(cross_bounds) >= 2:
        x = int(min(cross_bounds[0], cross_bounds[1]))
    elif cross_pixels:
        x = int(min(cross_pixels))
    else:
        x = 0
    return [x, y0, x, y1]


def _read_image_path(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _compose_image_with_grid_visual(source_img, grid_img):
    if source_img is None or grid_img is None or source_img.shape[:2] != grid_img.shape[:2]:
        return grid_img
    composed = source_img.copy()
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    grid_pixels = gray < 245
    if not np.any(grid_pixels):
        return composed
    composed[grid_pixels] = grid_img[grid_pixels]
    return composed


def _standard_native_grid_image(
    img,
    enhanced_x,
    enhanced_y,
    fallback_x_pixels,
    fallback_y_pixels,
    fallback_x_axis,
    fallback_y_axis,
    *,
    grid_line_color=None,
):
    x_info = enhanced_x if isinstance(enhanced_x, dict) else {}
    y_info = enhanced_y if isinstance(enhanced_y, dict) else {}
    x_pixels = x_info.get("native_pixels") or fallback_x_pixels or []
    y_pixels = y_info.get("native_pixels") or fallback_y_pixels or []
    if x_info or y_info:
        x_axis = _axis_line_from_enhanced_ticks("x", x_info, y_info, img.shape)
        y_axis = _axis_line_from_enhanced_ticks("y", y_info, x_info, img.shape)
    else:
        x_axis = fallback_x_axis
        y_axis = fallback_y_axis
    return draw_basic_grid(
        img,
        x_pixels,
        y_pixels,
        x_axis,
        y_axis,
        x_grid_bounds=x_info.get("bounds"),
        y_grid_bounds=y_info.get("bounds"),
        grid_line_color=grid_line_color or GRID_LINE_COLOR,
    )


def _try_fast_refresh_encrypted_grid_from_existing_outputs(img, image_path, output_dir, chart_id):
    if not _env_flag_enabled("CARTESIAN_REUSE_OUTPUT_ARTIFACTS", True):
        return None

    output_json_path = Path(output_dir) / f"{chart_id}_ticks.json"
    using_generic_sidecar = False
    if not output_json_path.exists():
        generic_sidecar_path = Path(output_dir) / "ticks.json"
        if generic_sidecar_path.exists():
            output_json_path = generic_sidecar_path
            using_generic_sidecar = True
        else:
            return None
    try:
        with output_json_path.open("r", encoding="utf-8") as file:
            tick_data = json.load(file)
    except Exception as exc:
        logger.warning("Existing tick sidecar cannot be reused: %s", exc)
        return None
    if not isinstance(tick_data, dict):
        return None

    enhanced_grid_reconstruction = tick_data.get("enhanced_grid_reconstruction")
    enhanced_encryption = _encrypted_ticks_from_enhanced_bindings(
        enhanced_grid_reconstruction,
        x_axis_scale=tick_data.get("x_axis_scale", "linear"),
        y_axis_scale=tick_data.get("y_axis_scale", "linear"),
    )
    if not enhanced_encryption:
        return None

    enhanced_x = enhanced_encryption.get("x")
    enhanced_y = enhanced_encryption.get("y")
    if not enhanced_x and not enhanced_y:
        return None

    x_ticks_encrypted = (enhanced_x or {}).get("encrypted_ticks") or tick_data.get("x_ticks_encrypted")
    y_ticks_encrypted = (enhanced_y or {}).get("encrypted_ticks") or tick_data.get("y_ticks_encrypted")
    x_pixels_encrypted = (enhanced_x or {}).get("encrypted_pixels") or tick_data.get("x_pixels_encrypted")
    y_pixels_encrypted = (enhanced_y or {}).get("encrypted_pixels") or tick_data.get("y_pixels_encrypted")
    x_pixels_data = (enhanced_x or {}).get("native_pixels") or tick_data.get("x_pixels")
    y_pixels_data = (enhanced_y or {}).get("native_pixels") or tick_data.get("y_pixels")
    has_x_encryption = bool(x_ticks_encrypted and x_pixels_encrypted)
    has_y_encryption = bool(y_ticks_encrypted and y_pixels_encrypted)
    if not (has_x_encryption or has_y_encryption):
        return None
    x_ticks_encrypted = x_ticks_encrypted or []
    y_ticks_encrypted = y_ticks_encrypted or []
    x_pixels_encrypted = x_pixels_encrypted or []
    y_pixels_encrypted = y_pixels_encrypted or []
    x_pixels_data = x_pixels_data or []
    y_pixels_data = y_pixels_data or []

    x_axis = _axis_line_from_enhanced_ticks("x", enhanced_x, enhanced_y, img.shape)
    y_axis = _axis_line_from_enhanced_ticks("y", enhanced_y, enhanced_x, img.shape)
    base_grid_img = _standard_native_grid_image(
        img,
        enhanced_x,
        enhanced_y,
        x_pixels_data,
        y_pixels_data,
        x_axis,
        y_axis,
        grid_line_color=GRID_LINE_COLOR,
    )
    if using_generic_sidecar:
        basic_grid_path = Path(output_dir) / "image_basic_grid.png"
    else:
        basic_grid_path = Path(tick_data.get("basic_grid_path") or Path(output_dir) / f"{chart_id}_grid.png")
    if base_grid_img is None or base_grid_img.shape[:2] != img.shape[:2]:
        return None
    success, encoded_basic_img = cv2.imencode(".png", base_grid_img)
    if success:
        basic_grid_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_basic_img.tofile(str(basic_grid_path))

    encrypted_grid_img = draw_encrypted_grid(
        img,
        x_pixels_encrypted,
        y_pixels_encrypted,
        x_ticks_encrypted,
        y_ticks_encrypted,
        x_axis,
        y_axis,
        x_axis_type=(enhanced_x or {}).get("axis_type", tick_data.get("x_axis_type", NUMERIC_AXIS_TYPE)),
        y_axis_type=(enhanced_y or {}).get("axis_type", tick_data.get("y_axis_type", NUMERIC_AXIS_TYPE)),
        base_x_pixels=x_pixels_data,
        base_y_pixels=y_pixels_data,
        base_grid_img=base_grid_img,
        x_grid_bounds=(enhanced_x or {}).get("bounds"),
        y_grid_bounds=(enhanced_y or {}).get("bounds"),
        x_label_styles=(enhanced_x or {}).get("encrypted_label_styles"),
        y_label_styles=(enhanced_y or {}).get("encrypted_label_styles"),
    )
    colored_grid_img = draw_encrypted_grid(
        img,
        x_pixels_encrypted,
        y_pixels_encrypted,
        x_ticks_encrypted,
        y_ticks_encrypted,
        x_axis,
        y_axis,
        x_axis_type=(enhanced_x or {}).get("axis_type", tick_data.get("x_axis_type", NUMERIC_AXIS_TYPE)),
        y_axis_type=(enhanced_y or {}).get("axis_type", tick_data.get("y_axis_type", NUMERIC_AXIS_TYPE)),
        base_x_pixels=x_pixels_data,
        base_y_pixels=y_pixels_data,
        base_grid_img=base_grid_img,
        x_grid_bounds=(enhanced_x or {}).get("bounds"),
        y_grid_bounds=(enhanced_y or {}).get("bounds"),
        x_label_styles=(enhanced_x or {}).get("encrypted_label_styles"),
        y_label_styles=(enhanced_y or {}).get("encrypted_label_styles"),
        grid_line_color=GRID_LINE_REVIEW_COLOR,
    )

    if using_generic_sidecar:
        encrypted_grid_path = Path(output_dir) / "image_with_grid.png"
        colored_grid_path = Path(output_dir) / "image_with_grid_color.png"
    else:
        encrypted_grid_path = Path(tick_data.get("encrypted_grid_path") or Path(output_dir) / f"{chart_id}_with_grid.png")
        colored_grid_path = encrypted_grid_path.with_name(f"{encrypted_grid_path.stem}_color{encrypted_grid_path.suffix}")
    encrypted_grid_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded_img = cv2.imencode(".png", encrypted_grid_img)
    if not success:
        return None
    encoded_img.tofile(str(encrypted_grid_path))
    success, encoded_color_img = cv2.imencode(".png", colored_grid_img)
    if success:
        encoded_color_img.tofile(str(colored_grid_path))

    if enhanced_x:
        tick_data["x_ticks"] = enhanced_x["native_ticks"]
        tick_data["x_pixels"] = enhanced_x["native_pixels"]
        tick_data["x_ticks_encrypted"] = enhanced_x["encrypted_ticks"]
        tick_data["x_pixels_encrypted"] = enhanced_x["encrypted_pixels"]
        tick_data["x_axis_type"] = enhanced_x["axis_type"]
        tick_data["x_label_orientation_angle"] = enhanced_x.get("stable_label_angle", 0.0)
        tick_data["x_skipped_encrypted_intervals"] = enhanced_x.get("skipped_encrypted_intervals", [])
        tick_data["x_dense_encrypted_interval_warnings"] = enhanced_x.get("dense_interval_warnings", [])
        tick_data["x_axis_encryption_policy"] = enhanced_x.get("axis_encryption_policy")
    if enhanced_y:
        tick_data["y_ticks"] = enhanced_y["native_ticks"]
        tick_data["y_pixels"] = enhanced_y["native_pixels"]
        tick_data["y_ticks_encrypted"] = enhanced_y["encrypted_ticks"]
        tick_data["y_pixels_encrypted"] = enhanced_y["encrypted_pixels"]
        tick_data["y_axis_type"] = enhanced_y["axis_type"]
        tick_data["y_label_orientation_angle"] = enhanced_y.get("stable_label_angle", 0.0)
        tick_data["y_skipped_encrypted_intervals"] = enhanced_y.get("skipped_encrypted_intervals", [])
        tick_data["y_dense_encrypted_interval_warnings"] = enhanced_y.get("dense_interval_warnings", [])
        tick_data["y_axis_encryption_policy"] = enhanced_y.get("axis_encryption_policy")
    tick_data["image_path"] = str(image_path)
    tick_data["basic_grid_path"] = str(basic_grid_path)
    tick_data["encrypted_grid_path"] = str(encrypted_grid_path)
    tick_data["colored_grid_path"] = str(colored_grid_path)
    tick_data["encrypted_label_style_version"] = ENCRYPTED_LABEL_STYLE_VERSION
    tick_data["fast_refresh"] = {
        "enabled": True,
        "source": "existing_tick_sidecar_and_enhanced_artifacts",
        "refreshed_encrypted_grid_path": str(encrypted_grid_path),
        "refreshed_colored_grid_path": str(colored_grid_path),
    }
    try:
        with output_json_path.open("w", encoding="utf-8") as file:
            json.dump(tick_data, file, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Failed to update fast-refreshed tick sidecar: %s", exc)
    logger.info("Fast-refreshed encrypted grid from existing artifacts: %s", encrypted_grid_path)
    return tick_data


def _process_chart_with_enhanced_grid_only(
    img,
    image_path,
    output_dir,
    chart_id,
    chart_type,
    axis_repair_hint=None,
    disable_cache=False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    enhanced_grid_reconstruction = _run_enhanced_cartesian_grid_reconstruction(
        image_path,
        output_dir,
        disable_cache=disable_cache,
    )
    if not isinstance(enhanced_grid_reconstruction, dict) or enhanced_grid_reconstruction.get("error"):
        logger.error(
            "Enhanced cartesian grid reconstruction failed in grid-first path: %s",
            (enhanced_grid_reconstruction or {}).get("error") if isinstance(enhanced_grid_reconstruction, dict) else None,
        )
        return None

    enhanced_encryption = _encrypted_ticks_from_enhanced_bindings(enhanced_grid_reconstruction)
    if not enhanced_encryption:
        logger.error("Enhanced cartesian grid reconstruction did not produce usable label bindings.")
        return None

    enhanced_x = enhanced_encryption.get("x")
    enhanced_y = enhanced_encryption.get("y")
    if not enhanced_x and not enhanced_y:
        logger.error("Enhanced cartesian grid reconstruction has no usable x/y axis bindings.")
        return None

    x_axis = _axis_line_from_enhanced_ticks("x", enhanced_x, enhanced_y, img.shape)
    y_axis = _axis_line_from_enhanced_ticks("y", enhanced_y, enhanced_x, img.shape)
    x_ticks_data = (enhanced_x or {}).get("native_ticks") or []
    y_ticks_data = (enhanced_y or {}).get("native_ticks") or []
    x_pixels_data = (enhanced_x or {}).get("native_pixels") or []
    y_pixels_data = (enhanced_y or {}).get("native_pixels") or []
    x_ticks_encrypted = (enhanced_x or {}).get("encrypted_ticks") or x_ticks_data
    y_ticks_encrypted = (enhanced_y or {}).get("encrypted_ticks") or y_ticks_data
    x_pixels_encrypted = (enhanced_x or {}).get("encrypted_pixels") or x_pixels_data
    y_pixels_encrypted = (enhanced_y or {}).get("encrypted_pixels") or y_pixels_data
    x_axis_type = (enhanced_x or {}).get("axis_type", TEXT_AXIS_TYPE if x_ticks_data else NUMERIC_AXIS_TYPE)
    y_axis_type = (enhanced_y or {}).get("axis_type", TEXT_AXIS_TYPE if y_ticks_data else NUMERIC_AXIS_TYPE)
    x_grid_bounds = (enhanced_x or {}).get("bounds")
    y_grid_bounds = (enhanced_y or {}).get("bounds")
    x_label_styles = (enhanced_x or {}).get("encrypted_label_styles")
    y_label_styles = (enhanced_y or {}).get("encrypted_label_styles")

    if not x_pixels_encrypted or not y_pixels_encrypted:
        logger.error("Enhanced cartesian grid reconstruction has empty encrypted pixel positions.")
        return None

    basic_grid_path = output_path / f"{chart_id}_grid.png"
    encrypted_grid_path = output_path / f"{chart_id}_with_grid.png"
    colored_grid_path = output_path / f"{chart_id}_with_grid_color.png"

    basic_grid_img = _standard_native_grid_image(
        img,
        enhanced_x,
        enhanced_y,
        x_pixels_data,
        y_pixels_data,
        x_axis,
        y_axis,
        grid_line_color=GRID_LINE_COLOR,
    )
    success, encoded_basic_img = cv2.imencode(".png", basic_grid_img)
    if success:
        encoded_basic_img.tofile(str(basic_grid_path))
    else:
        logger.error("Failed to encode enhanced-grid-first basic grid image: %s", basic_grid_path)
        return None

    encrypted_grid_img = draw_encrypted_grid(
        img,
        x_pixels_encrypted,
        y_pixels_encrypted,
        x_ticks_encrypted,
        y_ticks_encrypted,
        x_axis,
        y_axis,
        x_axis_type=x_axis_type,
        y_axis_type=y_axis_type,
        base_x_pixels=x_pixels_data,
        base_y_pixels=y_pixels_data,
        base_grid_img=basic_grid_img,
        x_grid_bounds=x_grid_bounds,
        y_grid_bounds=y_grid_bounds,
        x_label_styles=x_label_styles,
        y_label_styles=y_label_styles,
    )
    colored_grid_img = draw_encrypted_grid(
        img,
        x_pixels_encrypted,
        y_pixels_encrypted,
        x_ticks_encrypted,
        y_ticks_encrypted,
        x_axis,
        y_axis,
        x_axis_type=x_axis_type,
        y_axis_type=y_axis_type,
        base_x_pixels=x_pixels_data,
        base_y_pixels=y_pixels_data,
        base_grid_img=basic_grid_img,
        x_grid_bounds=x_grid_bounds,
        y_grid_bounds=y_grid_bounds,
        x_label_styles=x_label_styles,
        y_label_styles=y_label_styles,
        grid_line_color=GRID_LINE_REVIEW_COLOR,
    )

    success, encoded_encrypted_img = cv2.imencode(".png", encrypted_grid_img)
    if not success:
        logger.error("Failed to encode enhanced-grid-first encrypted grid image: %s", encrypted_grid_path)
        return None
    encoded_encrypted_img.tofile(str(encrypted_grid_path))
    success, encoded_colored_img = cv2.imencode(".png", colored_grid_img)
    if success:
        encoded_colored_img.tofile(str(colored_grid_path))

    colors_data = colors_from_axis_repair_hint(axis_repair_hint, chart_type)
    if not colors_data:
        colors_data = extract_chart_color_items(image_path, chart_type)
    series_color_data = series_color_from_items(colors_data)

    tick_data = {
        "chart_id": chart_id,
        "chart_type": chart_type,
        "x_ticks": x_ticks_data,
        "y_ticks": y_ticks_data,
        "x_pixels": x_pixels_data,
        "y_pixels": y_pixels_data,
        "x_ticks_encrypted": x_ticks_encrypted,
        "y_ticks_encrypted": y_ticks_encrypted,
        "x_pixels_encrypted": x_pixels_encrypted,
        "y_pixels_encrypted": y_pixels_encrypted,
        "x_axis_type": x_axis_type,
        "y_axis_type": y_axis_type,
        "x_axis_scale": "linear",
        "y_axis_scale": "linear",
        "x_label_orientation_angle": (enhanced_x or {}).get("stable_label_angle", 0.0),
        "y_label_orientation_angle": (enhanced_y or {}).get("stable_label_angle", 0.0),
        "x_skipped_encrypted_intervals": (enhanced_x or {}).get("skipped_encrypted_intervals", []),
        "y_skipped_encrypted_intervals": (enhanced_y or {}).get("skipped_encrypted_intervals", []),
        "x_dense_encrypted_interval_warnings": (enhanced_x or {}).get("dense_interval_warnings", []),
        "y_dense_encrypted_interval_warnings": (enhanced_y or {}).get("dense_interval_warnings", []),
        "x_axis_encryption_policy": (enhanced_x or {}).get("axis_encryption_policy"),
        "y_axis_encryption_policy": (enhanced_y or {}).get("axis_encryption_policy"),
        "image_path": str(image_path),
        "basic_grid_path": str(basic_grid_path),
        "encrypted_grid_path": str(encrypted_grid_path),
        "colored_grid_path": str(colored_grid_path),
        "encrypted_label_style_version": ENCRYPTED_LABEL_STYLE_VERSION,
        "generation_cache_disabled": bool(disable_cache),
        "colors": colors_data,
        "series_color": series_color_data,
        "axis_repair": {
            "hint": axis_repair_hint if isinstance(axis_repair_hint, dict) else {},
            "source": "enhanced_grid_first",
        },
        "enhanced_grid_reconstruction": enhanced_grid_reconstruction,
        "encrypted_tick_source": enhanced_encryption.get("source", "enhanced_grid_label_bindings"),
        "basic_grid_source": "standard_native_grid_from_enhanced_bindings",
        "cartesian_flow": "enhanced_grid_first",
        "legacy_cartesian_flow_used": False,
    }

    output_json_path = output_path / f"{chart_id}_ticks.json"
    try:
        with output_json_path.open("w", encoding="utf-8") as file:
            json.dump(tick_data, file, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Failed to write enhanced-grid-first tick sidecar: %s", exc)
    logger.info("Processed cartesian chart with enhanced-grid-first flow: %s", encrypted_grid_path)
    return tick_data


def normalize_axis_repair_hint(axis_repair_hint):
    hint = axis_repair_hint if isinstance(axis_repair_hint, dict) else {}

    def as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "missing"}
        return False

    def as_choice(value, choices, default="unknown"):
        text = str(value or "").strip().lower()
        return text if text in choices else default

    def normalize_tick_list(values, limit=40):
        if not isinstance(values, list):
            return []
        normalized = []
        for value in values[:limit]:
            if isinstance(value, (str, int, float)):
                text = str(value).strip()
                if text:
                    normalized.append(text)
        return normalized

    axis_tick_labels = hint.get("axis_tick_labels")
    if not isinstance(axis_tick_labels, dict):
        axis_tick_labels = {}
    series_items = normalize_series_items_hint(hint.get("series_items"))

    x_axis_missing = as_bool(hint.get("x_axis_missing", hint.get("x", False)))
    y_axis_missing = as_bool(hint.get("y_axis_missing", hint.get("y", False)))
    return {
        "x_axis_missing": x_axis_missing,
        "y_axis_missing": y_axis_missing,
        "x_ticks_missing": as_bool(hint.get("x_ticks_missing", False)) or x_axis_missing,
        "y_ticks_missing": as_bool(hint.get("y_ticks_missing", False)) or y_axis_missing,
        "x_axis_role": as_choice(hint.get("x_axis_role"), {"numeric", "category", "date", "unknown"}),
        "y_axis_role": as_choice(hint.get("y_axis_role"), {"numeric", "category", "date", "unknown"}),
        "x_axis_position": as_choice(
            hint.get("x_axis_position"),
            {"bottom", "top", "middle", "none", "unknown"},
        ),
        "y_axis_position": as_choice(
            hint.get("y_axis_position"),
            {"left", "right", "middle", "both", "none", "unknown"},
        ),
        "plot_area_style": as_choice(
            hint.get("plot_area_style"),
            {"explicit_axes", "weak_axes", "grid_only", "no_axes", "unknown"},
        ),
        "has_background_grid": as_bool(hint.get("has_background_grid", False)),
        "x_tick_recovery_from_grid": as_bool(hint.get("x_tick_recovery_from_grid", False)),
        "y_tick_recovery_from_grid": as_bool(hint.get("y_tick_recovery_from_grid", False)),
        "bar_layout": as_choice(
            hint.get("bar_layout"),
            {"single", "grouped", "stacked", "dense", "unknown"},
        ),
        "bar_orientation": as_choice(
            hint.get("bar_orientation"),
            {"vertical", "horizontal", "unknown"},
        ),
        "confidence": hint.get("confidence", 0),
        "reason": str(hint.get("reason", "") or ""),
        "axis_tick_labels": {
            "x_ticks": normalize_tick_list(axis_tick_labels.get("x_ticks")),
            "y_ticks": normalize_tick_list(axis_tick_labels.get("y_ticks")),
            "x_ticks_encrypted": normalize_tick_list(axis_tick_labels.get("x_ticks_encrypted"), limit=80),
            "y_ticks_encrypted": normalize_tick_list(axis_tick_labels.get("y_ticks_encrypted"), limit=80),
            "x_axis_type": as_choice(
                axis_tick_labels.get("x_axis_type", hint.get("x_axis_role")),
                {"numeric", "category", "date", "unknown"},
            ),
            "y_axis_type": as_choice(
                axis_tick_labels.get("y_axis_type", hint.get("y_axis_role")),
                {"numeric", "category", "date", "unknown"},
            ),
            "source": str(axis_tick_labels.get("source", "upload_detection") or "upload_detection"),
        },
        "series_items": series_items,
    }


def normalize_series_items_hint(value):
    if not isinstance(value, dict):
        return {"items": [], "source": "unknown", "kind": "unknown"}
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        raw_items = value.get("legend_items")
    if not isinstance(raw_items, list):
        raw_items = value.get("point_items")
    if not isinstance(raw_items, list):
        raw_items = []

    items = []
    seen = set()
    for raw_item in raw_items[:80]:
        if not isinstance(raw_item, dict):
            continue
        name = str(raw_item.get("name") or raw_item.get("label") or "").strip()
        if not name:
            continue
        color = raw_item.get("color")
        if isinstance(color, str):
            color = color.strip()
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                color = None
        else:
            color = None
        key = (name.casefold(), color or "")
        if key in seen:
            continue
        seen.add(key)
        items.append({"name": name, "color": color})

    return {
        "items": items,
        "source": str(value.get("source") or "unknown"),
        "kind": str(value.get("kind") or "unknown").strip().lower() or "unknown",
    }


def colors_from_axis_repair_hint(axis_repair_hint, chart_type=None):
    hint = axis_repair_hint if isinstance(axis_repair_hint, dict) else {}
    series_items = normalize_series_items_hint(hint.get("series_items"))
    colors = []
    for item in series_items.get("items", []):
        color = item.get("color")
        if not isinstance(color, str) or not color:
            continue
        name = str(item.get("name") or f"Series {len(colors) + 1}").strip() or f"Series {len(colors) + 1}"
        colors.append({"name": name, "color": color})
    return colors


def extract_chart_color_items(image_path, chart_type):
    try:
        if chart_type in {"scatter", "bubble"}:
            colors_result = extract_point_chart_items(image_path)
        else:
            colors_result = extract_chart_series_color(image_path)
    except Exception as error:
        logger.warning("Failed to extract chart color items: %s", error)
        colors_result = []
    if isinstance(colors_result, list):
        colors_data = colors_result
    elif colors_result:
        colors_data = [{"name": "Series 1", "color": str(colors_result)}]
    else:
        colors_data = []
    result = []
    for index, item in enumerate(colors_data):
        if not isinstance(item, dict) or not item.get("color"):
            continue
        name = str(item.get("name") or f"Series {index + 1}").strip() or f"Series {index + 1}"
        result.append({"name": name, "color": str(item["color"])})
    return result


def series_color_from_items(colors_data):
    result = {}
    if not isinstance(colors_data, list):
        return result
    for index, item in enumerate(colors_data):
        if not isinstance(item, dict) or not item.get("color"):
            continue
        name = str(item.get("name") or f"Series {index + 1}").strip() or f"Series {index + 1}"
        result[name] = str(item["color"])
    return result


def axis_repair_enabled(axis_repair_hint):
    hint = normalize_axis_repair_hint(axis_repair_hint)
    return any(
        hint.get(key)
        for key in (
            "x_axis_missing",
            "y_axis_missing",
            "x_ticks_missing",
            "y_ticks_missing",
            "x_tick_recovery_from_grid",
            "y_tick_recovery_from_grid",
        )
    ) or hint.get("plot_area_style") in {"weak_axes", "grid_only", "no_axes"}


def _append_hint_reason(hint, reason):
    if not reason:
        return hint
    old_reason = str(hint.get("reason", "") or "")
    hint["reason"] = f"{old_reason} | {reason}" if old_reason else reason
    return hint


def _mark_missing_axis_from_cv(chart_type, axis_repair_hint, x_axis, y_axis, boxes):
    """Turn CV failures into repair hints without changing recognized chart type."""
    hint = normalize_axis_repair_hint(axis_repair_hint)
    chart_type = (chart_type or "").lower()
    if chart_type in {"scatter", "bubble"}:
        if x_axis is None:
            hint["x_axis_missing"] = True
            hint["x_ticks_missing"] = True
            _append_hint_reason(hint, "CV could not locate point-chart x-axis")
        if y_axis is None:
            hint["y_axis_missing"] = True
            hint["y_ticks_missing"] = True
            _append_hint_reason(hint, "CV could not locate point-chart y-axis")
        if hint.get("plot_area_style") in {"weak_axes", "grid_only", "no_axes"}:
            hint["x_tick_recovery_from_grid"] = bool(hint.get("has_background_grid"))
            hint["y_tick_recovery_from_grid"] = bool(hint.get("has_background_grid"))
            _append_hint_reason(hint, f"MLLM plot style: {hint.get('plot_area_style')}")
        return hint

    if not _is_bar_chart_type(chart_type) or not boxes:
        return hint
    if x_axis is None:
        hint["x_axis_missing"] = True
        hint["x_ticks_missing"] = True
        _append_hint_reason(hint, "CV could not locate x-axis")
    if y_axis is None:
        hint["y_axis_missing"] = True
        hint["y_ticks_missing"] = True
        _append_hint_reason(hint, "CV could not locate y-axis")
    if hint.get("plot_area_style") in {"weak_axes", "grid_only", "no_axes"}:
        if _bar_orientation(chart_type) == "v":
            hint["x_ticks_missing"] = True
            hint["y_tick_recovery_from_grid"] = bool(hint.get("has_background_grid"))
        elif _bar_orientation(chart_type) == "h":
            hint["y_ticks_missing"] = True
            hint["x_tick_recovery_from_grid"] = bool(hint.get("has_background_grid"))
        _append_hint_reason(hint, f"MLLM plot style: {hint.get('plot_area_style')}")
    return hint


def _axis_needs_grid_recovery(axis_repair_hint, direction):
    hint = normalize_axis_repair_hint(axis_repair_hint)
    return (
        hint.get(f"{direction}_tick_recovery_from_grid")
        or (
            hint.get("has_background_grid")
            and hint.get("plot_area_style") in {"weak_axes", "grid_only"}
            and hint.get(f"{direction}_ticks_missing")
        )
    )


def _point_chart_grid_hint(axis_repair_hint):
    hint = normalize_axis_repair_hint(axis_repair_hint)
    return bool(hint.get("has_background_grid")) or hint.get("plot_area_style") in {
        "grid_only",
        "weak_axes",
        "no_axes",
    }


def _clamp_axis_line(axis, image_shape):
    if axis is None:
        return None
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in axis]
    return [
        int(np.clip(x1, 0, w - 1)),
        int(np.clip(y1, 0, h - 1)),
        int(np.clip(x2, 0, w - 1)),
        int(np.clip(y2, 0, h - 1)),
    ]


def _line_len(line):
    return float(np.hypot(line[2] - line[0], line[3] - line[1]))


# ===== 基础坐标轴候选提取 =====
# 这组函数从 Hough 合并后的线段里补充寻找水平/垂直轴。
# 它们主要作为 infer_axes_from_lines 失败后的保底，不直接决定所有图表的最终轴。
def _horizontal_axis_from_lines(lines, w, h):
    best = None
    best_score = float("-inf")
    for line in lines or []:
        x1, y1, x2, y2 = [int(v) for v in line]
        if abs(y1 - y2) > 8:
            continue
        length = abs(x2 - x1)
        if length < max(30, w * 0.25):
            continue
        y = int(round((y1 + y2) / 2))
        score = length + y * 0.8
        if score > best_score:
            best_score = score
            best = [min(x1, x2), y, max(x1, x2), y]
    return best


def _vertical_axis_from_lines(lines, w, h):
    best = None
    best_score = float("-inf")
    for line in lines or []:
        x1, y1, x2, y2 = [int(v) for v in line]
        if abs(x1 - x2) > 8:
            continue
        length = abs(y2 - y1)
        if length < max(30, h * 0.25):
            continue
        x = int(round((x1 + x2) / 2))
        score = length + (w - x) * 0.35
        if score > best_score:
            best_score = score
            best = [x, max(y1, y2), x, min(y1, y2)]
    return best


def _bar_orientation(chart_type):
    chart_type = (chart_type or "").lower()
    if chart_type in {"h_bar", "h_stacked_bar"}:
        return "h"
    if chart_type in {"v_bar", "v_stacked_bar"}:
        return "v"
    return ""


def _is_bar_chart_type(chart_type):
    return _bar_orientation(chart_type) in {"h", "v"}


def _bar_base_type(chart_type):
    orientation = _bar_orientation(chart_type)
    if orientation == "h":
        return "h_bar"
    if orientation == "v":
        return "v_bar"
    return (chart_type or "").lower()


def _bar_boxes(img, chart_type):
    orientation = _bar_orientation(chart_type)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([179, 255, 255]))
    dark_mask = cv2.inRange(gray, 0, 235)
    light_grid_mask = cv2.inRange(gray, 180, 235) & cv2.inRange(hsv[:, :, 1], 0, 35)
    mask = cv2.bitwise_or(color_mask, dark_mask)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(light_grid_mask))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    min_area = max(6, w * h * 0.00008)
    boxes = []
    dense_plot_boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if orientation == "v" and bw >= w * 0.25 and bh >= h * 0.15 and y + bh <= h * 0.99:
            dense_plot_boxes.append((int(x), int(y), int(bw), int(bh)))
        elif orientation == "h" and bw >= w * 0.15 and bh >= h * 0.25 and x + bw <= w * 0.99:
            dense_plot_boxes.append((int(x), int(y), int(bw), int(bh)))
        if area < min_area:
            continue
        if orientation == "h":
            if bw < 12 or bh < 2 or bw < bh * 2:
                continue
        elif orientation == "v":
            if bh < 10 or bw < 1 or bh < bw * 1.5:
                continue
        else:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))
    if orientation == "h":
        if len(boxes) >= 3:
            median_height = float(np.median([box[3] for box in boxes]))
            boxes = [box for box in boxes if box[3] >= max(4, median_height * 0.45)]
        if dense_plot_boxes and (
            not boxes
            or len(boxes) < 3
            or float(np.median([box[2] for box in boxes])) < max(16.0, w * 0.08)
        ):
            return sorted([max(dense_plot_boxes, key=lambda box: box[2] * box[3])], key=lambda box: box[1])
        return sorted(boxes, key=lambda box: box[1])
    if len(boxes) >= 3:
        median_width = float(np.median([box[2] for box in boxes]))
        boxes = [box for box in boxes if box[2] >= max(1, median_width * 0.45)]
        if len(boxes) >= 3:
            bottoms = [box[1] + box[3] for box in boxes]
            baseline = float(np.percentile(bottoms, 85))
            median_height = float(np.median([box[3] for box in boxes]))
            tolerance = max(8.0, median_height * 0.35)
            baseline_boxes = [box for box in boxes if abs((box[1] + box[3]) - baseline) <= tolerance]
            if len(baseline_boxes) >= max(2, int(len(boxes) * 0.45)):
                boxes = baseline_boxes
    if dense_plot_boxes and (
        not boxes
        or len(boxes) < 3
        or float(np.median([box[3] for box in boxes])) < max(16.0, h * 0.08)
    ):
        return sorted([max(dense_plot_boxes, key=lambda box: box[2] * box[3])], key=lambda box: box[0])
    return sorted(boxes, key=lambda box: box[0])


def _axis_y(axis):
    return int(round((axis[1] + axis[3]) / 2))


def _axis_x(axis):
    return int(round((axis[0] + axis[2]) / 2))


def _dedupe_pixels(pixels, tolerance=4):
    values = sorted(int(round(pixel)) for pixel in pixels)
    groups = []
    for value in values:
        if not groups or abs(value - groups[-1][-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [int(round(float(np.median(group)))) for group in groups]


def _tick_group_count(ticks, direction):
    if not ticks:
        return 0
    if direction == "x":
        pixels = [(tick[0] + tick[2]) / 2 for tick in ticks]
    else:
        pixels = [(tick[1] + tick[3]) / 2 for tick in ticks]
    return len(_dedupe_pixels(pixels))


def _is_bar_category_axis(chart_type, direction):
    direction = (direction or "").lower()
    orientation = _bar_orientation(chart_type)
    return (orientation == "h" and direction == "y") or (
        orientation == "v" and direction == "x"
    )


def _required_tick_count(chart_type, direction):
    return 1 if _is_bar_category_axis(chart_type, direction) else 2


def _is_placeholder_category_tick(value):
    return bool(re.fullmatch(r"category_\d+", str(value or "").strip().casefold()))


def _drop_mixed_placeholder_category_ticks(tick_values, pixel_positions, chart_type, direction):
    """Remove synthetic category_N ticks only when real category labels exist too.

    Grouped bar charts can expose legend marker rows/columns as extra detected
    category ticks. If the MLLM already returned real category labels, keeping
    both creates fake prediction targets and draws grid lines in the legend.
    """
    if not _is_bar_category_axis(chart_type, direction):
        return tick_values, pixel_positions, False
    if not tick_values or not pixel_positions or len(tick_values) != len(pixel_positions):
        return tick_values, pixel_positions, False

    placeholders = [_is_placeholder_category_tick(value) for value in tick_values]
    if not any(placeholders) or all(placeholders):
        return tick_values, pixel_positions, False

    filtered = [
        (tick, pixel)
        for tick, pixel, is_placeholder in zip(tick_values, pixel_positions, placeholders)
        if not is_placeholder
    ]
    if len(filtered) < _required_tick_count(chart_type, direction):
        return tick_values, pixel_positions, False

    clean_ticks, clean_pixels = zip(*filtered)
    logger.debug(
        "Dropped %s mixed placeholder %s category ticks for %s: %s -> %s",
        sum(placeholders),
        direction,
        chart_type,
        tick_values,
        list(clean_ticks),
    )
    return list(clean_ticks), list(clean_pixels), True


def _drop_secondary_group_labels_for_bar_axis(chart_type, direction, tick_values):
    if not _is_bar_category_axis(chart_type, direction) or len(tick_values or []) < 6:
        return tick_values, False
    values = [str(value).strip() for value in tick_values]
    year_like = [bool(re.fullmatch(r"(?:19|20)\d{2}", value)) for value in values]
    if not any(year_like):
        return tick_values, False
    primary = [value for value, is_year in zip(values, year_like) if not is_year]
    if len(primary) < 4:
        return tick_values, False
    has_repeated_primary = len(primary) > len(set(primary))
    # Keep pure year axes intact. Only remove years when another repeated
    # category level exists, e.g. Q1 Q2 Q3 Q4 grouped under 2017, 2018...
    if not has_repeated_primary or len(primary) < sum(year_like) * 2:
        return tick_values, False
    return primary, True


def _horizontal_gridline_plot_span(lines, x_axis, image_shape):
    if x_axis is None:
        return None

    h, w = image_shape[:2]
    axis_y = _axis_y(x_axis)
    axis_left, axis_right = sorted([int(x_axis[0]), int(x_axis[2])])
    candidates = []
    min_length = max(40, (axis_right - axis_left) * 0.45, w * 0.2)
    for line in lines or []:
        x1, y1, x2, y2 = [int(value) for value in line]
        if abs(y1 - y2) > 4:
            continue
        left, right = sorted([x1, x2])
        length = right - left
        if length < min_length:
            continue
        y = int(round((y1 + y2) / 2))
        if y < h * 0.05 or y > axis_y + max(8, h * 0.01):
            continue
        candidates.append((left, right, y))

    if not candidates:
        return None

    left = int(round(float(np.median([item[0] for item in candidates]))))
    right = int(np.percentile([item[1] for item in candidates], 90))
    y_pixels = _dedupe_pixels([item[2] for item in candidates])
    if axis_y not in y_pixels:
        y_pixels.append(axis_y)
    y_pixels = sorted(_dedupe_pixels(y_pixels), reverse=True)
    return left, right, y_pixels


def _vbar_y_axis_side(axis_repair_hint):
    hint = normalize_axis_repair_hint(axis_repair_hint)
    return "right" if hint.get("y_axis_position") == "right" else "left"


def _vbar_axes_from_bounds(plot_left, plot_right, plot_top, plot_bottom, axis_repair_hint):
    plot_left = int(round(plot_left))
    plot_right = int(round(plot_right))
    plot_top = int(round(plot_top))
    plot_bottom = int(round(plot_bottom))
    y_axis_x = plot_right if _vbar_y_axis_side(axis_repair_hint) == "right" else plot_left
    return (
        [plot_left, plot_bottom, plot_right, plot_bottom],
        [y_axis_x, plot_bottom, y_axis_x, plot_top],
    )


def repair_missing_axes(img, merged_lines, x_axis, y_axis, chart_type, axis_repair_hint):
    # 根据 MLLM/CV 的缺轴提示补出轴线。
    # bar 图优先使用柱体包围盒；v_bar 支持右侧 Y 轴；
    # scatter/bubble 的复杂网格修复会在后续点图专用函数中继续细化。
    hint = normalize_axis_repair_hint(axis_repair_hint)
    if not axis_repair_enabled(hint):
        return x_axis, y_axis, []

    chart_type = (chart_type or "").lower()
    h, w = img.shape[:2]
    boxes = _bar_boxes(img, chart_type)
    if (
        _bar_orientation(chart_type) == "h"
        and h > 600
        and hint.get("plot_area_style") in {"no_axes", "weak_axes"}
    ):
        plot_boxes = [box for box in boxes if box[1] >= h * 0.12]
        if len(plot_boxes) >= max(3, int(len(boxes) * 0.55)):
            boxes = plot_boxes

    if x_axis is None and not hint["x_axis_missing"]:
        x_axis = _horizontal_axis_from_lines(merged_lines, w, h)
    if y_axis is None and not hint["y_axis_missing"]:
        y_axis = _vertical_axis_from_lines(merged_lines, w, h)

    if not boxes:
        return x_axis, y_axis, boxes

    left = min(box[0] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    top = min(box[1] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)

    orientation = _bar_orientation(chart_type)
    if orientation == "h":
        if hint["y_axis_missing"]:
            bottom_y = _axis_y(x_axis) if x_axis is not None else min(h - 1, bottom + max(8, int(np.median([b[3] for b in boxes]))))
            top_y = max(0, top)
            y_axis = [left, bottom_y, left, top_y]
        if hint["x_axis_missing"]:
            axis_x = _axis_x(y_axis) if y_axis is not None else left
            axis_y = max(y_axis[1], y_axis[3]) if y_axis is not None else min(h - 1, bottom + 8)
            x_axis = [axis_x, axis_y, right, axis_y]

    elif orientation == "v":
        if hint["x_axis_missing"]:
            axis_y = max(box[1] + box[3] for box in boxes)
            axis_x = _axis_x(y_axis) if y_axis is not None else left
            x_axis = [axis_x, axis_y, right, axis_y]
        if hint["y_axis_missing"]:
            grid_span = _horizontal_gridline_plot_span(merged_lines, x_axis, img.shape)
            axis_y = _axis_y(x_axis) if x_axis is not None else bottom
            if grid_span:
                grid_left, grid_right, grid_y_pixels = grid_span
                if (
                    len(boxes) == 1
                    and boxes[0][2] > w * 0.35
                    and grid_left < left
                    and left - grid_left <= max(30, int(w * 0.08))
                ):
                    grid_left = left
                axis_x = grid_right if _vbar_y_axis_side(hint) == "right" else grid_left
                top_y = min(grid_y_pixels)
                if x_axis is not None:
                    x_axis = [grid_left, axis_y, max(grid_right, x_axis[2], x_axis[0]), axis_y]
            else:
                axis_x = (
                    max(x_axis[0], x_axis[2])
                    if x_axis is not None and _vbar_y_axis_side(hint) == "right"
                    else min(x_axis[0], x_axis[2]) if x_axis is not None else left
                )
                top_y = top
            y_axis = [axis_x, axis_y, axis_x, top_y]

    return x_axis, y_axis, boxes


def _synthetic_bar_tick_pixels(chart_type, direction, boxes):
    if not boxes:
        return []
    orientation = _bar_orientation(chart_type)
    if orientation == "h" and direction == "y":
        return sorted([int(round(y + h / 2)) for _, y, _, h in boxes], reverse=True)
    if orientation == "v" and direction == "x":
        return sorted([int(round(x + w / 2)) for x, _, w, _ in boxes])
    return []


def _cluster_bar_category_pixels(pixels, target_count, direction):
    """Group multiple bars into category centers when labels imply grouped bars."""
    pixels = sorted(_dedupe_pixels(pixels or []))
    if target_count is None or target_count <= 0 or len(pixels) <= target_count:
        return sorted(pixels, reverse=(direction == "y"))

    gaps = [
        (pixels[index + 1] - pixels[index], index)
        for index in range(len(pixels) - 1)
    ]
    split_indices = {
        index
        for _, index in sorted(gaps, reverse=True)[: max(0, target_count - 1)]
    }
    groups = []
    current = [pixels[0]]
    for index, pixel in enumerate(pixels[1:]):
        if index in split_indices:
            groups.append(current)
            current = [pixel]
        else:
            current.append(pixel)
    groups.append(current)

    if len(groups) != target_count or any(not group for group in groups):
        centers = np.linspace(0, len(pixels) - 1, target_count)
        result = [pixels[int(round(center))] for center in centers]
    else:
        result = [int(round(float(np.median(group)))) for group in groups]
    return sorted(_dedupe_pixels(result), reverse=(direction == "y"))


def synthesize_tick_pixels_for_missing_axis(
    chart_type,
    direction,
    axis,
    boxes,
    tick_values,
    axis_repair_hint,
):
    # 当真实短 tick 不可见时，基于图表类型生成候选 tick 像素：
    # - bar 类别轴：用柱体中心，分组柱会聚类成类别中心。
    # - 数值轴：在轴线范围内按 tick 文本数量等距生成。
    hint = normalize_axis_repair_hint(axis_repair_hint)
    if not (hint.get(f"{direction}_axis_missing") or hint.get(f"{direction}_ticks_missing")):
        return []

    chart_type = (chart_type or "").lower()
    bar_pixels = _synthetic_bar_tick_pixels(chart_type, direction, boxes)
    if bar_pixels:
        if _is_bar_category_axis(chart_type, direction) and len(tick_values or []) >= 1:
            if len(bar_pixels) < len(tick_values):
                bar_pixels = []
            else:
                clustered = _cluster_bar_category_pixels(
                    bar_pixels,
                    len(tick_values),
                    direction,
                )
                if clustered:
                    return clustered
        if bar_pixels:
            return bar_pixels

    if axis is None or len(tick_values or []) < 2:
        return []

    count = len(tick_values)
    if direction == "x":
        start, end = sorted([int(axis[0]), int(axis[2])])
    else:
        low, high = sorted([int(axis[1]), int(axis[3])])
        start, end = high, low
    return [int(round(value)) for value in np.linspace(start, end, count)]


def _finite_positive_bar_values(values):
    numeric = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return []
        if math.isfinite(number) and number > 0:
            numeric.append(number)
    return numeric


def _finite_bar_values(values, allow_negative=False):
    numeric = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        if not allow_negative and number <= 0:
            continue
        if allow_negative and abs(number) <= 1e-9:
            continue
        numeric.append(number)
    return numeric


def _nice_numeric_ticks_from_data_labels(values, max_count=7):
    numeric = _finite_positive_bar_values(values)
    if not numeric:
        return []
    max_value = max(numeric)
    if max_value <= 0:
        return []
    raw_step = max_value / max(3, max_count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    candidates = [1, 2, 2.5, 5, 10]
    step = candidates[-1] * magnitude
    for candidate in candidates:
        candidate_step = candidate * magnitude
        if math.floor(max_value / candidate_step) + 1 <= max_count:
            step = candidate_step
            break
    if step <= 0:
        return []
    ticks = [0.0]
    value = step
    while value <= max_value * 1.001 and len(ticks) < max_count:
        ticks.append(round(value, 10))
        value += step
    if len(ticks) < 3:
        ticks = [0.0, round(max_value / 2, 10), round(max_value, 10)]
    return ticks


def _nice_diverging_ticks_from_data_labels(values, max_count=7):
    numeric = _finite_bar_values(values, allow_negative=True)
    if not numeric:
        return []
    min_value = min(numeric)
    max_value = max(numeric)
    span = max_value - min_value
    if span <= 0:
        return []
    raw_step = span / max(3, max_count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = 10 * magnitude
    for candidate in [1, 2, 2.5, 5, 10]:
        candidate_step = candidate * magnitude
        low = math.floor(min_value / candidate_step) * candidate_step
        high = math.ceil(max_value / candidate_step) * candidate_step
        span_count = int(round((high - low) / candidate_step)) + 1
        if span_count <= max_count:
            step = candidate_step
            break
    low = math.floor(min_value / step) * step
    high = math.ceil(max_value / step) * step
    ticks = []
    value = low
    while value <= high + step * 0.1:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _bar_measurement_boxes_for_value_labels(chart_type, direction, axis, boxes):
    orientation = _bar_orientation(chart_type)
    if axis is None:
        return boxes or []
    if orientation == "v" and direction == "y":
        baseline = _axis_y(axis)
        candidates = [
            box for box in boxes or []
            if abs((box[1] + box[3]) - baseline) <= max(5, box[3] * 0.12)
        ]
        return candidates or (boxes or [])
    if orientation == "h" and direction == "x":
        baseline = _axis_x(axis)
        candidates = [
            box for box in boxes or []
            if box[0] - 3 <= baseline <= box[0] + box[2] + 3
        ]
        return candidates or (boxes or [])
    return boxes or []


def _bar_value_axis_ticks_from_data_labels(chart_type, direction, axis, boxes, values):
    # 柱形图缺少数值轴 tick 时的兜底：
    # MLLM 先读取柱端/柱顶的数据标签，这里再结合柱体长度或高度，
    # 反推出一组可用于加密的数值轴 tick 和像素位置。
    """Use bar-end data labels as anchors when a value axis has no visible ticks."""
    orientation = _bar_orientation(chart_type)
    if not boxes or axis is None:
        return [], []
    if (orientation == "v" and direction != "y") or (orientation == "h" and direction != "x"):
        return [], []

    boxes = _bar_measurement_boxes_for_value_labels(chart_type, direction, axis, boxes)
    numeric = _finite_bar_values(values, allow_negative=(orientation == "h"))
    pair_count = min(len(numeric), len(boxes or []))
    if pair_count < 2:
        return [], []
    numeric_for_fit = numeric[:pair_count]
    boxes_for_fit = (boxes or [])[:pair_count]

    if orientation == "v":
        ordered_boxes = sorted(boxes_for_fit, key=lambda box: box[0] + box[2] / 2)
        baseline = _axis_y(axis)
        anchors = []
        for value, box in zip(numeric_for_fit, ordered_boxes):
            top = int(box[1])
            span = baseline - top
            if span > 2:
                anchors.append(span / value)
        if len(anchors) < 2:
            return [], []
        scale = float(np.median(anchors))
        tick_values = _nice_numeric_ticks_from_data_labels(numeric)
        tick_pixels = [int(round(baseline - value * scale)) for value in tick_values]
    else:
        ordered_boxes = sorted(boxes_for_fit, key=lambda box: box[1] + box[3] / 2)
        baseline = _axis_x(axis)
        anchors = []
        for value, box in zip(numeric_for_fit, ordered_boxes):
            left, right = int(box[0]), int(box[0] + box[2])
            if value >= 0:
                end = right
            else:
                end = left
            span = abs(end - baseline)
            if span > 2:
                anchors.append(span / abs(value))
        if len(anchors) < 2:
            return [], []
        scale = float(np.median(anchors))
        tick_values = (
            _nice_diverging_ticks_from_data_labels(numeric)
            if min(numeric) < 0 < max(numeric)
            else _nice_numeric_ticks_from_data_labels(numeric)
        )
        tick_pixels = [int(round(baseline + value * scale)) for value in tick_values]

    paired = [(value, pixel) for value, pixel in zip(tick_values, tick_pixels)]
    if len(paired) < 2:
        return [], []
    tick_values, tick_pixels = zip(*paired)
    if direction == "y":
        pairs = sorted(zip(tick_values, tick_pixels), key=lambda item: item[0])
    else:
        pairs = sorted(zip(tick_values, tick_pixels), key=lambda item: item[0])
    return [value for value, _ in pairs], [pixel for _, pixel in pairs]


def ticks_from_pixels(pixels, axis, direction):
    if axis is None:
        return []
    ticks = []
    if direction == "x":
        y = _axis_y(axis)
        for x in pixels:
            ticks.append([int(x), y + 1, int(x), y + 6])
    else:
        x = _axis_x(axis)
        for y in pixels:
            ticks.append([x - 6, int(y), x - 1, int(y)])
    return ticks


# ===== 从背景网格恢复 tick 像素 =====
# 当图表没有短 tick，或短 tick 被文字/点/图例干扰时，先从 Hough 网格线
# 或浅灰色网格投影中找候选像素，后续再用 MLLM 读到的 tick 数值反选正确组合。
def infer_tick_pixels_from_gridlines(lines, x_axis, y_axis, direction, image_shape):
    """Infer tick positions from full plot gridlines when short tick marks are absent."""
    if x_axis is None or y_axis is None:
        return []

    h, w = image_shape[:2]
    axis_y = _axis_y(x_axis)
    axis_x = _axis_x(y_axis)
    plot_left = max(0, min(axis_x, min(x_axis[0], x_axis[2])))
    plot_right = min(w - 1, max(x_axis[0], x_axis[2]))
    plot_top = max(0, min(y_axis[1], y_axis[3]))
    plot_bottom = min(h - 1, max(axis_y, y_axis[1], y_axis[3]))
    plot_width = max(1, plot_right - plot_left)
    plot_height = max(1, plot_bottom - plot_top)
    border_margin_x = max(6, int(round(plot_width * 0.015)))
    border_margin_y = max(6, int(round(plot_height * 0.03)))

    pixels = []
    for line in lines or []:
        x1, y1, x2, y2 = [int(v) for v in line]
        if direction == "y":
            if abs(y1 - y2) > 4:
                continue
            y = int(round((y1 + y2) / 2))
            if y <= plot_top + border_margin_y or y >= plot_bottom - border_margin_y:
                continue
            left, right = sorted([x1, x2])
            overlap = max(0, min(right, plot_right) - max(left, plot_left))
            if overlap >= plot_width * 0.35:
                pixels.append(y)
        else:
            if abs(x1 - x2) > 4:
                continue
            x = int(round((x1 + x2) / 2))
            if x <= plot_left + border_margin_x or x >= plot_right - border_margin_x:
                continue
            top, bottom = sorted([y1, y2])
            overlap = max(0, min(bottom, plot_bottom) - max(top, plot_top))
            if overlap >= plot_height * 0.35:
                pixels.append(x)

    if not pixels:
        return []

    pixels = sorted(set(int(p) for p in pixels))
    if direction == "y":
        pixels = sorted(pixels, reverse=True)
    return pixels


def _group_projection_peaks(scores, *, min_score=0.08, max_gap=3):
    peak_indices = [index for index, score in enumerate(scores) if score >= min_score]
    if not peak_indices:
        return []

    groups = []
    for index in peak_indices:
        if not groups or index - groups[-1][-1] > max_gap:
            groups.append([index])
        else:
            groups[-1].append(index)

    peaks = []
    for group in groups:
        weights = np.array([float(scores[index]) for index in group])
        if float(weights.sum()) <= 0:
            center = int(round(sum(group) / len(group)))
        else:
            center = int(round(float(np.average(group, weights=weights))))
        peaks.append((center, float(max(weights)), len(group)))
    return peaks


def infer_point_chart_grid_pixels_by_projection(img, x_axis, y_axis, direction, expected_count=None):
    # scatter/bubble/line 的弱轴图常常没有短 tick，但有浅灰色背景网格。
    # 这里通过像素投影找出竖向/横向网格线中心，作为 tick 像素候选。
    """Detect faint dashed plot gridlines by projecting light gray pixels.

    This is a fallback for point charts where Hough-based tick detection locks on
    to bubble/text fragments instead of the real dashed grid. It deliberately
    looks only inside the plot area and is used only after the ordinary tick
    result is found to be suspicious.
    """
    if img is None or x_axis is None or y_axis is None:
        return []

    h, w = img.shape[:2]
    axis_x = _axis_x(y_axis)
    axis_y = _axis_y(x_axis)
    plot_top = max(0, min(int(y_axis[1]), int(y_axis[3])))
    plot_bottom = min(h - 1, max(axis_y, int(y_axis[1]), int(y_axis[3])))
    if plot_bottom - plot_top < max(40, h * 0.15):
        return []

    b, g, r = cv2.split(img)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    gray_grid_mask = (maxc - minc <= 12) & (maxc >= 180) & (maxc <= 245)

    direction = (direction or "").lower()
    if direction == "x":
        y1 = max(0, plot_top)
        y2 = min(h, plot_bottom + 1)
        x1 = max(0, axis_x - 12)
        x2 = min(w, int(round(w * 0.88)))
        if y2 <= y1 or x2 <= x1:
            return []
        scores = gray_grid_mask[y1:y2, x1:x2].mean(axis=0)
        if len(scores) >= 3:
            scores = np.convolve(scores, np.ones(3) / 3, mode="same")
        peaks = _group_projection_peaks(scores, min_score=0.08, max_gap=3)
        pixels = [x1 + center for center, score, width in peaks if width <= 10]
        pixels = sorted(_dedupe_pixels(pixels))
    elif direction == "y":
        x_grid = infer_point_chart_grid_pixels_by_projection(
            img, x_axis, y_axis, "x", expected_count=None
        )
        x1 = min(x_grid) if len(x_grid) >= 2 else max(0, axis_x - 4)
        x2 = max(x_grid) if len(x_grid) >= 2 else min(w - 1, max(int(x_axis[0]), int(x_axis[2])))
        x1 = max(0, x1)
        x2 = min(w, x2 + 1)
        y1 = max(0, plot_top)
        y2 = min(h, plot_bottom - max(18, int(round((plot_bottom - plot_top) * 0.06))))
        if y2 <= y1 or x2 <= x1:
            return []
        scores = gray_grid_mask[y1:y2, x1:x2].mean(axis=1)
        if len(scores) >= 3:
            scores = np.convolve(scores, np.ones(3) / 3, mode="same")
        peaks = _group_projection_peaks(scores, min_score=0.08, max_gap=3)
        pixels = [y1 + center for center, score, width in peaks if width <= 12]
        pixels = sorted(_dedupe_pixels(pixels), reverse=True)
    else:
        return []

    return pixels


def infer_point_chart_grid_pixels_for_missing_axes(img, direction):
    """Infer candidate plot gridlines without trusting detected axis lines.

    Real-world point charts often omit both axis strokes while keeping a light
    background grid. This projection is only used inside the missing-axis repair
    path, so regular dataset charts with visible axes keep the established flow.
    """
    if img is None:
        return []

    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    gray_grid_mask = (maxc - minc <= 12) & (maxc >= 180) & (maxc <= 245)

    direction = (direction or "").lower()
    if direction == "x":
        x1 = max(0, int(round(w * 0.04)))
        x2 = min(w, int(round(w * 0.88)))
        y1 = max(0, int(round(h * 0.20)))
        y2 = min(h, int(round(h * 0.80)))
        if y2 <= y1 or x2 <= x1:
            return []
        scores = gray_grid_mask[y1:y2, x1:x2].mean(axis=0)
        if len(scores) >= 3:
            scores = np.convolve(scores, np.ones(3) / 3, mode="same")
        peaks = _group_projection_peaks(scores, min_score=0.075, max_gap=3)
        pixels = [x1 + center for center, score, width in peaks if width <= 12]
        return sorted(_dedupe_pixels(pixels))

    if direction == "y":
        x_candidates = infer_point_chart_grid_pixels_for_missing_axes(img, "x")
        if len(x_candidates) >= 2:
            x1 = max(0, min(x_candidates) - 2)
            x2 = min(w, max(x_candidates) + 3)
        else:
            x1 = max(0, int(round(w * 0.04)))
            x2 = min(w, int(round(w * 0.88)))
        y1 = max(0, int(round(h * 0.20)))
        y2 = min(h, int(round(h * 0.82)))
        if y2 <= y1 or x2 <= x1:
            return []
        scores = gray_grid_mask[y1:y2, x1:x2].mean(axis=1)
        if len(scores) >= 3:
            scores = np.convolve(scores, np.ones(3) / 3, mode="same")
        peaks = _group_projection_peaks(scores, min_score=0.075, max_gap=3)
        pixels = [y1 + center for center, score, width in peaks if width <= 14]
        return sorted(_dedupe_pixels(pixels), reverse=True)

    return []


def _finite_numeric_sequence(values):
    numeric = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        numeric.append(number)
    return numeric


def _fit_tick_pixels(values, pixels, scale):
    numeric = np.array(values, dtype=float)
    candidate_pixels = np.array(pixels, dtype=float)
    if len(numeric) != len(candidate_pixels) or len(numeric) < 2:
        return float("inf")
    if scale == "log":
        if np.any(numeric <= 0):
            return float("inf")
        numeric = np.log10(numeric)
    try:
        coeff = np.polyfit(numeric, candidate_pixels, 1)
        predicted = np.polyval(coeff, numeric)
    except Exception:
        return float("inf")
    span = max(1.0, float(candidate_pixels.max() - candidate_pixels.min()))
    return float(np.sqrt(np.mean((predicted - candidate_pixels) ** 2)) / span)


def _values_prefer_log_scale(values):
    numeric = _finite_numeric_sequence(values)
    if not numeric or len(numeric) < 3 or any(value <= 0 for value in numeric):
        return False
    linear_gaps = np.diff(numeric)
    log_gaps = np.diff(np.log10(numeric))
    if np.any(linear_gaps == 0) or np.any(log_gaps == 0):
        return False
    linear_cv = float(np.std(np.abs(linear_gaps)) / max(1e-9, np.mean(np.abs(linear_gaps))))
    log_cv = float(np.std(np.abs(log_gaps)) / max(1e-9, np.mean(np.abs(log_gaps))))
    return linear_cv > 0.35 and log_cv < max(0.35, linear_cv * 0.55)


def _snap_scaled_ticks_to_candidates(pixels, numeric, direction, scale):
    """Bind ticks to visual candidates by projecting values onto candidate endpoints."""
    ordered = sorted(_dedupe_pixels(pixels or []), reverse=(direction == "y"))
    if len(ordered) < len(numeric or []) or len(numeric or []) < 2:
        return None

    values = np.array(numeric, dtype=float)
    if scale == "log":
        if np.any(values <= 0):
            return None
        values = np.log10(values)
    if not np.all(np.diff(values) > 0) and not np.all(np.diff(values) < 0):
        return None

    denom = float(values[-1] - values[0])
    if abs(denom) <= 1e-12:
        return None
    ratios = (values - values[0]) / denom

    total_span = abs(float(max(ordered) - min(ordered)))
    if total_span <= 0:
        return None
    gaps = [abs(ordered[index + 1] - ordered[index]) for index in range(len(ordered) - 1)]
    positive_gaps = [gap for gap in gaps if gap > 0]
    median_gap = float(np.median(positive_gaps)) if positive_gaps else max(1.0, total_span / max(1, len(ordered) - 1))
    snap_tolerance = max(4.0, min(18.0, total_span * 0.035, median_gap * 0.55))
    min_endpoint_span = total_span * (0.40 if len(numeric) <= 4 else 0.55)

    best = None
    max_endpoint_pairs = 8000
    endpoint_pairs = 0
    for start_index in range(0, len(ordered) - len(numeric) + 1):
        start = float(ordered[start_index])
        for end_index in range(start_index + len(numeric) - 1, len(ordered)):
            end = float(ordered[end_index])
            endpoint_span = abs(end - start)
            if endpoint_span < min_endpoint_span:
                continue
            endpoint_pairs += 1
            if endpoint_pairs > max_endpoint_pairs:
                break
            expected = start + ratios * (end - start)
            snapped = []
            used = set()
            total_error = 0.0
            ok = True
            for expected_pixel in expected:
                available = [candidate for candidate in ordered if candidate not in used]
                if not available:
                    ok = False
                    break
                nearest = min(available, key=lambda candidate: abs(float(candidate) - float(expected_pixel)))
                error = abs(float(nearest) - float(expected_pixel))
                if error > snap_tolerance:
                    ok = False
                    break
                used.add(nearest)
                snapped.append(int(nearest))
                total_error += error * error
            if not ok or len(snapped) != len(numeric):
                continue
            if direction == "y":
                if any(snapped[index] <= snapped[index + 1] for index in range(len(snapped) - 1)):
                    continue
            elif any(snapped[index] >= snapped[index + 1] for index in range(len(snapped) - 1)):
                continue
            score = math.sqrt(total_error / len(snapped)) / max(1.0, endpoint_span)
            fit_score = _fit_tick_pixels(numeric, snapped, scale)
            score = max(score, fit_score)
            if (
                best is None
                or score < best[0] - 1e-4
                or (abs(score - best[0]) <= 1e-4 and endpoint_span > best[3])
            ):
                best = (score, scale, snapped, endpoint_span)
        if endpoint_pairs > max_endpoint_pairs:
            break
    return best


def _axis_endpoint_candidates(direction, axis):
    if axis is None:
        return []
    try:
        if direction == "x":
            return sorted([int(round(axis[0])), int(round(axis[2]))])
        if direction == "y":
            return sorted([int(round(axis[1])), int(round(axis[3]))], reverse=True)
    except Exception:
        return []
    return []


def select_projected_tick_pixels_for_values(projected_pixels, tick_values, direction, axis=None):
    # 从一堆候选网格线里选择与 MLLM tick 数值最匹配的一组。
    # 这里会同时尝试 linear/log 两种尺度，并通过拟合误差、端点覆盖范围、
    # 单调性来避免“tick 整体错位一个”的情况。
    """Choose the gridline subset that best matches the visible tick labels."""
    numeric = _finite_numeric_sequence(tick_values)
    if not numeric or len(numeric) < 2:
        return [], "linear"

    direction = (direction or "").lower()
    pixels = sorted(
        _dedupe_pixels(list(projected_pixels or []) + _axis_endpoint_candidates(direction, axis)),
        reverse=(direction == "y"),
    )
    target_count = len(numeric)
    if len(pixels) < target_count:
        return [], "linear"
    if len(pixels) == target_count:
        scale = axis_scale_from_ticks_and_pixels(numeric, pixels)
        return pixels, scale

    scales = ["linear"]
    if _values_prefer_log_scale(numeric):
        scales.insert(0, "log")

    best = None
    for scale in scales:
        scaled = _snap_scaled_ticks_to_candidates(pixels, numeric, direction, scale)
        if scaled is None:
            continue
        if best is None or scaled[0] < best[0] - 1e-4 or (
            abs(scaled[0] - best[0]) <= 1e-4 and scaled[3] > best[3]
        ):
            best = scaled
    if best is not None and best[0] <= 0.075:
        return best[2], best[1]

    total_combinations = math.comb(len(pixels), target_count)
    if total_combinations > 20000:
        # Keep the search bounded while preserving plot extremes and quantiles.
        quantile_indices = {
            int(round(value))
            for value in np.linspace(0, len(pixels) - 1, min(len(pixels), target_count * 2 + 6))
        }
        pixels = [pixels[index] for index in sorted(quantile_indices)]
        total_combinations = math.comb(len(pixels), target_count)
    if total_combinations > 20000:
        return (best[2], best[1]) if best is not None else ([], "linear")

    for combo in itertools.combinations(pixels, target_count):
        for scale in scales:
            score = _fit_tick_pixels(numeric, combo, scale)
            span = abs(float(max(combo) - min(combo)))
            if (
                best is None
                or score < best[0] - 1e-4
                or (abs(score - best[0]) <= 1e-4 and span > best[3])
            ):
                best = (score, scale, list(combo), span)

    if best is None:
        return [], "linear"
    return best[2], best[1]


def axis_scale_from_ticks_and_pixels(tick_values, pixel_positions):
    # 根据 tick 数值序列和像素间距判断当前数值轴更像线性轴还是对数轴。
    # 后续加密 tick 的数值和像素插值都会使用这个尺度。
    numeric = _finite_numeric_sequence(tick_values)
    if not numeric or len(numeric) < 3 or len(pixel_positions or []) != len(numeric):
        return "linear"
    if not _values_prefer_log_scale(numeric):
        return "linear"
    linear_score = _fit_tick_pixels(numeric, pixel_positions, "linear")
    log_score = _fit_tick_pixels(numeric, pixel_positions, "log")
    if log_score < linear_score * 0.65:
        return "log"
    return "linear"


def _gray_grid_mask(img):
    b, g, r = cv2.split(img)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    return (maxc - minc <= 12) & (maxc >= 180) & (maxc <= 245)


def _mask_groups(values):
    groups = []
    for value in values:
        value = int(value)
        if not groups or value - groups[-1][-1] > 3:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [(group[0], group[-1], len(group)) for group in groups if len(group) >= 2]


def _regular_dash_span(groups):
    if len(groups) < 6:
        return None
    centers = [int(round((left + right) / 2)) for left, right, _ in groups]
    best = []
    for start_index in range(len(groups)):
        seq = [start_index]
        last_center = centers[start_index]
        for index in range(start_index + 1, len(groups)):
            gap = centers[index] - last_center
            if 5 <= gap <= 11:
                seq.append(index)
                last_center = centers[index]
            elif gap > 16:
                break
        if len(seq) > len(best):
            best = seq
    if len(best) < 6:
        return None
    return groups[best[0]][0], groups[best[-1]][1]


def _regular_dash_left_in_window(groups, min_left, max_left):
    centers = [int(round((left + right) / 2)) for left, right, _ in groups]
    for start_index, (left, _, _) in enumerate(groups):
        if left < min_left or left > max_left:
            continue
        seq_len = 1
        last_center = centers[start_index]
        for index in range(start_index + 1, len(groups)):
            gap = centers[index] - last_center
            if 5 <= gap <= 11:
                seq_len += 1
                last_center = centers[index]
            elif gap > 16:
                break
        if seq_len >= 6:
            return int(left)
    return None


def infer_point_chart_plot_bounds_from_horizontal_grid(img, y_pixels, x_pixels):
    """Estimate plot left/right from the dashed horizontal grid, not tick pixels."""
    if img is None or len(y_pixels or []) < 2:
        return None
    h, w = img.shape[:2]
    mask = _gray_grid_mask(img)
    spans = []
    left_edges = []
    tick_min = min(x_pixels or [0])
    tick_max = max(x_pixels or [w - 1])
    min_left = max(0, int(round(w * 0.04)))
    max_left = max(min_left, tick_min - max(35, int(round(w * 0.05))))
    for y in y_pixels:
        y = int(round(y))
        band = mask[max(0, y - 2) : min(h, y + 3), :]
        if band.size == 0:
            continue
        scores = band.mean(axis=0)
        groups = _mask_groups(np.where(scores >= 0.2)[0])
        left_edge = _regular_dash_left_in_window(groups, min_left, max_left)
        if left_edge is not None:
            left_edges.append(left_edge)
        span = _regular_dash_span(groups)
        if span is not None:
            spans.append(span)
    if not spans and not left_edges:
        return None

    left_candidates = [
        left for left, right in spans
        if left < tick_min - max(20, int(round(w * 0.03)))
    ]
    right_candidates = [
        right for left, right in spans
        if right > tick_max + max(10, int(round(w * 0.015)))
    ]
    plot_left = int(round(float(np.median(left_edges or left_candidates or [min(left for left, _ in spans)]))))
    plot_right = int(round(float(np.median(right_candidates or [max(right for _, right in spans)]))))
    if plot_right <= plot_left:
        return None
    return plot_left, plot_right


def _vertical_grid_centers_near_x(mask, x_pixels, y_low, y_high):
    h, w = mask.shape[:2]
    centers = []
    for x in x_pixels or []:
        x = int(round(x))
        band = mask[:, max(0, x - 2) : min(w, x + 3)]
        if band.size == 0:
            continue
        scores = band.mean(axis=1)
        groups = _mask_groups(np.where(scores >= 0.2)[0])
        for top, bottom, width in groups:
            center = int(round((top + bottom) / 2))
            if y_low <= center <= y_high and width <= 10:
                centers.append(center)
    return centers


def _supported_center_near_expected(centers, expected, tolerance=7):
    if not centers:
        return None
    groups = []
    for center in sorted(int(c) for c in centers):
        if not groups or center - groups[-1][-1] > tolerance:
            groups.append([center])
        else:
            groups[-1].append(center)
    groups = [group for group in groups if len(group) >= 3]
    if not groups:
        return None
    best = min(groups, key=lambda group: abs(float(np.median(group)) - expected))
    return int(round(float(np.median(best))))


def infer_point_chart_plot_vertical_bounds_from_grid(img, y_pixels, x_pixels):
    """Estimate plot top/bottom from vertical grid extent around selected x ticks."""
    if img is None or len(y_pixels or []) < 2 or len(x_pixels or []) < 2:
        return None
    sorted_y = sorted(int(round(value)) for value in y_pixels)
    gaps = [sorted_y[index + 1] - sorted_y[index] for index in range(len(sorted_y) - 1)]
    gaps = [gap for gap in gaps if gap > 4]
    if not gaps:
        return None
    gap = float(np.median(gaps))
    top_tick = min(sorted_y)
    bottom_tick = max(sorted_y)
    expected_top = top_tick - gap
    expected_bottom = bottom_tick + gap
    mask = _gray_grid_mask(img)

    top_centers = _vertical_grid_centers_near_x(
        mask,
        x_pixels,
        max(0, int(round(top_tick - gap * 1.35))),
        int(round(top_tick - gap * 0.25)),
    )
    bottom_centers = _vertical_grid_centers_near_x(
        mask,
        x_pixels,
        int(round(bottom_tick + gap * 0.25)),
        min(mask.shape[0] - 1, int(round(bottom_tick + gap * 1.25))),
    )
    plot_top = _supported_center_near_expected(top_centers, expected_top)
    plot_bottom = _supported_center_near_expected(bottom_centers, expected_bottom)
    if plot_top is None and plot_bottom is None:
        return None
    plot_top = int(plot_top if plot_top is not None else top_tick)
    plot_bottom = int(plot_bottom if plot_bottom is not None else bottom_tick)
    if plot_bottom <= plot_top:
        return None
    return plot_top, plot_bottom


def refine_point_chart_axes_from_projected_ticks(
    img,
    x_axis,
    y_axis,
    x_pixel_positions,
    y_pixel_positions,
    y_axis_scale="linear",
    prefer_grid_bounds=False,
):
    # 点图轴线修复：当 tick 像素已经从背景网格中选出来后，
    # 反过来用这些 tick 位置确定完整绘图区边界和 X/Y 轴线。
    """Place repaired point-chart axes using selected gridline ticks.

    For OWID-style bubble charts, the lowest visible y tick is often the plot
    baseline. Extrapolating one more grid step below it places the synthetic
    x-axis in the x-axis title/label band, so keep linear y axes on the lowest
    selected y tick. Log-scaled y axes may include an unlabeled lower plot
    boundary, so they still use the vertical-grid extent when available.
    """
    if img is None or len(x_pixel_positions or []) < 2 or len(y_pixel_positions or []) < 2:
        return x_axis, y_axis, False

    plot_bounds = infer_point_chart_plot_bounds_from_horizontal_grid(
        img,
        y_pixel_positions,
        x_pixel_positions,
    )
    if plot_bounds:
        plot_left, plot_right = plot_bounds
        plot_right = max(int(plot_right), int(max(x_pixel_positions)))
    else:
        plot_left = int(min(x_pixel_positions))
        plot_right = int(max(x_pixel_positions))

    vertical_bounds = infer_point_chart_plot_vertical_bounds_from_grid(
        img,
        y_pixel_positions,
        x_pixel_positions,
    )
    if str(y_axis_scale or "linear").lower() == "linear":
        plot_top = int(min(y_pixel_positions))
        plot_bottom = int(max(y_pixel_positions))
    elif vertical_bounds:
        plot_top, plot_bottom = vertical_bounds
    else:
        plot_top = int(min(y_pixel_positions))
        plot_bottom = int(max(y_pixel_positions))

    if plot_right <= plot_left or plot_bottom <= plot_top:
        return x_axis, y_axis, False

    return (
        [plot_left, plot_bottom, plot_right, plot_bottom],
        [plot_left, plot_bottom, plot_left, plot_top],
        True,
    )


def point_chart_tick_pixels_are_suspicious(pixel_positions, axis, direction, expected_count):
    if axis is None or expected_count is None or expected_count < 2:
        return False
    if len(pixel_positions or []) < 2:
        return True

    if direction == "x":
        axis_span = abs(int(axis[2]) - int(axis[0]))
        pixel_span = max(pixel_positions) - min(pixel_positions)
    else:
        axis_span = abs(int(axis[3]) - int(axis[1]))
        pixel_span = max(pixel_positions) - min(pixel_positions)

    if axis_span <= 0:
        return False
    if len(pixel_positions) != expected_count and pixel_span < axis_span * 0.85:
        return True
    return pixel_span < axis_span * 0.35


def recover_bar_value_axis_pixels_from_grid(
    img,
    merged_lines,
    x_axis,
    y_axis,
    chart_type,
    direction,
    tick_values,
):
    # bar 数值轴缺短 tick 时，优先从可见背景网格线中恢复数值轴 tick 像素。
    # 它和 select_projected_tick_pixels_for_values 配合，保证像素位置与 tick 数值一致。
    """Recover bar value-axis tick pixels from visible plot grid lines."""
    if not _is_bar_chart_type(chart_type) or len(tick_values or []) < 2:
        return [], "linear"

    direction = (direction or "").lower()
    if (chart_type in {"v_bar", "v_stacked_bar"} and direction != "y") or (
        chart_type in {"h_bar", "h_stacked_bar"} and direction != "x"
    ):
        return [], "linear"

    projected = infer_tick_pixels_from_gridlines(
        merged_lines,
        x_axis,
        y_axis,
        direction,
        img.shape,
    )
    if len(projected) < len(tick_values or []):
        projected = infer_point_chart_grid_pixels_by_projection(
            img,
            x_axis,
            y_axis,
            direction,
            expected_count=len(tick_values or []),
        )
    selected, scale = select_projected_tick_pixels_for_values(
        projected,
        tick_values,
        direction,
        x_axis if direction == "x" else y_axis,
    )
    if len(selected) == len(tick_values or []) and len(selected) >= 2:
        return selected, scale
    return [], "linear"


def bootstrap_point_chart_tick_pixels(img, merged_lines, x_axis, y_axis, direction, min_count=2):
    """Provide provisional point-chart tick pixels when short tick marks are absent.

    The returned pixels are only a bridge to the MLLM tick-label step. Later
    code rebinds them to the actual tick values using grid projection.
    """
    if img is None or x_axis is None or y_axis is None:
        return []

    direction = (direction or "").lower()
    pixels = infer_tick_pixels_from_gridlines(
        merged_lines,
        x_axis,
        y_axis,
        direction,
        img.shape,
    )
    if len(pixels) < min_count:
        pixels = infer_point_chart_grid_pixels_by_projection(
            img,
            x_axis,
            y_axis,
            direction,
            expected_count=None,
        )
    if len(pixels) < min_count:
        pixels = infer_point_chart_grid_pixels_for_missing_axes(img, direction)
    if len(pixels) < min_count:
        axis = x_axis if direction == "x" else y_axis
        pixels = _linspace_pixels_on_axis(direction, axis, 5)

    return sorted(_dedupe_pixels(pixels), reverse=(direction == "y"))


def refine_point_chart_axes_for_bootstrap(img, x_axis, y_axis):
    """Use visible background grid bounds before tick detection when available."""
    if img is None or x_axis is None or y_axis is None:
        return x_axis, y_axis, False
    x_pixels = infer_point_chart_grid_pixels_for_missing_axes(img, "x")
    y_pixels = infer_point_chart_grid_pixels_for_missing_axes(img, "y")
    if len(x_pixels) < 2 or len(y_pixels) < 2:
        return x_axis, y_axis, False
    return refine_point_chart_axes_from_projected_ticks(
        img,
        x_axis,
        y_axis,
        x_pixels,
        y_pixels,
        "linear",
        prefer_grid_bounds=True,
    )


def infer_point_plot_bounds_from_gray_horizontal_grid(img):
    # 真实 scatter/bubble 常见浅灰色横向虚线网格。
    # 这里从灰色网格的连续 span 中估计 plot left/right/top/bottom。
    """Recover plot bounds from long light-gray horizontal gridlines."""
    if img is None:
        return None
    h, w = img.shape[:2]
    mask = _gray_grid_mask(img)
    y1 = max(0, int(round(h * 0.08)))
    y2 = min(h, int(round(h * 0.88)))
    if y2 <= y1:
        return None

    row_scores = mask[y1:y2, :].mean(axis=1)
    if len(row_scores) >= 3:
        row_scores = np.convolve(row_scores, np.ones(3) / 3, mode="same")
    row_peaks = _group_projection_peaks(row_scores, min_score=0.14, max_gap=3)
    y_pixels = []
    spans = []
    for center, score, width in row_peaks:
        y = y1 + center
        if width > 10:
            continue
        band = mask[max(0, y - 2) : min(h, y + 3), :]
        if band.size == 0:
            continue
        col_scores = band.mean(axis=0)
        groups = _mask_groups(np.where(col_scores >= 0.20)[0])
        groups = [(left, right, length) for left, right, length in groups if right - left >= w * 0.18]
        if not groups:
            continue
        left, right, _ = max(groups, key=lambda item: item[2])
        y_pixels.append(y)
        spans.append((left, right))

    if len(y_pixels) < 3 or not spans:
        return None

    left = int(min(item[0] for item in spans))
    right = int(max(item[1] for item in spans))
    top = int(min(y_pixels))
    bottom = int(max(y_pixels))
    if right - left < w * 0.25 or bottom - top < h * 0.18:
        return None
    return left, right, top, bottom, sorted(_dedupe_pixels(y_pixels), reverse=True)


def infer_point_axes_from_grid_candidates(img, merged_lines):
    """Recover point-chart plot bounds from Hough gridlines plus projections."""
    if img is None:
        return None
    h, w = img.shape[:2]
    projected_x = infer_point_chart_grid_pixels_for_missing_axes(img, "x")
    projected_y = infer_point_chart_grid_pixels_for_missing_axes(img, "y")

    verticals = []
    horizontals = []
    for line in merged_lines or []:
        x1, y1, x2, y2 = [int(value) for value in line]
        if abs(x1 - x2) <= 4:
            top, bottom = sorted([y1, y2])
            if bottom - top >= h * 0.35 and top <= h * 0.35 and bottom >= h * 0.55:
                verticals.append((int(round((x1 + x2) / 2)), top, bottom))
        elif abs(y1 - y2) <= 4:
            left, right = sorted([x1, x2])
            if right - left >= w * 0.22:
                y = int(round((y1 + y2) / 2))
                if h * 0.08 <= y <= h * 0.90:
                    horizontals.append((left, right, y))

    x_candidates = _regular_grid_pixels(
        [item[0] for item in verticals] + list(projected_x),
        min_count=3,
    )
    y_candidates = _regular_grid_pixels(
        [item[2] for item in horizontals] + list(projected_y),
        min_count=3,
    )
    if len(x_candidates) < 2 or (len(y_candidates) < 2 and not verticals):
        return None

    left = int(min(x_candidates))
    right_candidates = [max(x_candidates)]
    if horizontals:
        plausible_rights = [
            right for left_edge, right, _ in horizontals
            if right >= max(x_candidates) - max(10, int(w * 0.015))
        ]
        if plausible_rights:
            right_candidates.append(int(np.percentile(plausible_rights, 80)))
    right = int(max(right_candidates))

    if verticals:
        relevant_verticals = [
            item for item in verticals
            if min(x_candidates) - 4 <= item[0] <= max(x_candidates) + 4
        ] or verticals
        top = int(round(float(np.median([item[1] for item in relevant_verticals]))))
        bottom = int(round(float(np.median([item[2] for item in relevant_verticals]))))
    else:
        top = int(min(y_candidates))
        bottom = int(max(y_candidates))

    if y_candidates:
        top = min(top, int(min(y_candidates)))
        bottom = max(bottom, int(max(y_candidates)))

    if right - left < w * 0.20 or bottom - top < h * 0.18:
        return None
    return [left, bottom, right, bottom], [left, bottom, left, top]


def infer_point_tick_pixels_from_grid_candidates(img, merged_lines, direction):
    """Infer point-chart tick pixels from regular Hough/projection grid candidates."""
    if img is None:
        return []
    h, w = img.shape[:2]
    direction = (direction or "").lower()
    projected = infer_point_chart_grid_pixels_for_missing_axes(img, direction)

    verticals = []
    horizontals = []
    for line in merged_lines or []:
        x1, y1, x2, y2 = [int(value) for value in line]
        if abs(x1 - x2) <= 4:
            top, bottom = sorted([y1, y2])
            if bottom - top >= h * 0.35 and top <= h * 0.35 and bottom >= h * 0.55:
                verticals.append((int(round((x1 + x2) / 2)), top, bottom))
        elif abs(y1 - y2) <= 4:
            left, right = sorted([x1, x2])
            if right - left >= w * 0.22:
                y = int(round((y1 + y2) / 2))
                if h * 0.08 <= y <= h * 0.90:
                    horizontals.append((left, right, y))

    if direction == "x":
        pixels = _regular_grid_pixels(
            [item[0] for item in verticals] + list(projected),
            min_count=3,
        )
        return sorted(_dedupe_pixels(pixels))

    if direction == "y":
        candidates = [item[2] for item in horizontals] + list(projected)
        if verticals:
            vertical_top = int(round(float(np.median([item[1] for item in verticals]))))
            vertical_bottom = int(round(float(np.median([item[2] for item in verticals]))))
            margin = max(8, int(round(h * 0.02)))
            candidates = [
                value for value in candidates
                if vertical_top - margin <= int(value) <= vertical_bottom + margin
            ]
        pixels = sorted(_dedupe_pixels(candidates))
        if len(pixels) >= 2 and verticals:
            gaps = [pixels[index + 1] - pixels[index] for index in range(len(pixels) - 1)]
            gaps = [gap for gap in gaps if gap > 8]
            if gaps:
                median_gap = float(np.median(gaps))
                tolerance = max(8.0, median_gap * 0.30)
                top = int(round(float(np.median([item[1] for item in verticals]))))
                bottom = int(round(float(np.median([item[2] for item in verticals]))))
                if pixels[0] - top > 0 and abs((pixels[0] - top) - median_gap) <= tolerance:
                    pixels.insert(0, top)
                if bottom - pixels[-1] > 0 and abs((bottom - pixels[-1]) - median_gap) <= tolerance:
                    pixels.append(bottom)
        pixels = _regular_grid_pixels(pixels, min_count=3)
        return sorted(_dedupe_pixels(pixels), reverse=True)

    return []


# 点图缺轴或弱轴时的总入口：按优先级尝试灰色网格、Hough 网格候选、
# 投影网格和点云边界，尽量恢复完整绘图区轴线。
def infer_point_axes_from_visual_structure(img, merged_lines, chart_type, x_axis=None, y_axis=None, axis_repair_hint=None):
    """Synthesize scatter/bubble axes from background gridlines or point extent.

    This is intentionally gated to point charts whose axes are missing or whose
    MLLM attributes say the plot uses weak/background-grid axes. Normal dataset
    charts with explicit axes keep the ordinary Hough-based path.
    """
    chart_type = (chart_type or "").lower()
    if img is None or chart_type not in {"scatter", "bubble"}:
        return x_axis, y_axis, False
    hint = normalize_axis_repair_hint(axis_repair_hint)
    needs_repair = (
        x_axis is None
        or y_axis is None
        or _point_chart_grid_hint(hint)
        or hint.get("x_tick_recovery_from_grid")
        or hint.get("y_tick_recovery_from_grid")
    )
    if not needs_repair:
        return x_axis, y_axis, False

    h, w = img.shape[:2]
    gray_grid_bounds = infer_point_plot_bounds_from_gray_horizontal_grid(img)
    if gray_grid_bounds:
        left, right, top, bottom, _ = gray_grid_bounds
        return [left, bottom, right, bottom], [left, bottom, left, top], True

    grid_axes = infer_point_axes_from_grid_candidates(img, merged_lines)
    if grid_axes:
        return grid_axes[0], grid_axes[1], True

    x_pixels = infer_point_chart_grid_pixels_for_missing_axes(img, "x")
    y_pixels = infer_point_chart_grid_pixels_for_missing_axes(img, "y")
    if len(x_pixels) >= 2 and len(y_pixels) >= 2:
        left = int(min(x_pixels))
        right = int(max(x_pixels))
        top = int(min(y_pixels))
        bottom = int(max(y_pixels))
        if right - left > w * 0.25 and bottom - top > h * 0.20:
            return [left, bottom, right, bottom], [left, bottom, left, top], True

    bbox = _saturated_point_bbox(img)
    if bbox:
        left, top, right, bottom = bbox
        pad_x = max(6, int(round((right - left) * 0.05)))
        pad_y = max(6, int(round((bottom - top) * 0.08)))
        left = max(0, left - pad_x)
        right = min(w - 1, right + pad_x)
        top = max(0, top - pad_y)
        bottom = min(h - 1, bottom + pad_y)
        if right - left > w * 0.25 and bottom - top > h * 0.20:
            return [left, bottom, right, bottom], [left, bottom, left, top], True

    return x_axis, y_axis, False


def refine_explicit_point_axes_from_strokes(img, x_axis, y_axis, chart_type, axis_repair_hint):
    """Extend explicit scatter/bubble axes when Hough returns a shortened stroke."""
    chart_type = (chart_type or "").lower()
    if (
        img is None
        or chart_type not in {"scatter", "bubble"}
        or x_axis is None
        or y_axis is None
        or _point_chart_grid_hint(axis_repair_hint)
    ):
        return x_axis, y_axis, False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    dark = gray < 120
    changed = False

    def merge_stroke_groups(groups, max_gap):
        merged = []
        for left, right, length in groups:
            if not merged or left - merged[-1][1] > max_gap:
                merged.append([left, right, length])
            else:
                merged[-1][1] = right
                merged[-1][2] += length
        return [(left, right, length) for left, right, length in merged]

    x1, x2 = sorted([int(x_axis[0]), int(x_axis[2])])
    y0 = int(np.clip(_axis_y(x_axis), 0, h - 1))
    x_band = dark[max(0, y0 - 3) : min(h, y0 + 4), :]
    if x_band.size:
        scores = x_band.mean(axis=0)
        raw_groups = [group for group in _mask_groups(np.where(scores >= 0.10)[0]) if group[2] >= 8]
        current_span = max(1, x2 - x1)
        groups = merge_stroke_groups(raw_groups, max(12, int(current_span * 0.12)))
        candidates = []
        for left, right, length in groups:
            overlap = max(0, min(right, x2) - max(left, x1))
            if overlap >= min(current_span * 0.45, 80) and (right - left) > current_span * 1.08:
                candidates.append((left, right, length))
        if candidates:
            left, right, _ = max(candidates, key=lambda item: item[2])
            if right - left > current_span:
                x_axis = [int(x1), y0, int(right), y0]
                changed = True

    y1, y2 = sorted([int(y_axis[1]), int(y_axis[3])])
    x0 = int(np.clip(_axis_x(y_axis), 0, w - 1))
    y_band = dark[:, max(0, x0 - 3) : min(w, x0 + 4)]
    if y_band.size:
        scores = y_band.mean(axis=1)
        raw_groups = [group for group in _mask_groups(np.where(scores >= 0.10)[0]) if group[2] >= 8]
        current_span = max(1, y2 - y1)
        groups = merge_stroke_groups(raw_groups, max(12, int(current_span * 0.12)))
        candidates = []
        for top, bottom, length in groups:
            overlap = max(0, min(bottom, y2) - max(top, y1))
            if overlap >= min(current_span * 0.45, 80) and (bottom - top) > current_span * 1.08:
                candidates.append((top, bottom, length))
        if candidates:
            top, bottom, _ = max(candidates, key=lambda item: item[2])
            if bottom - top > current_span:
                y_axis = [x0, int(y2), x0, int(min(top, y1))]
                changed = True

    return x_axis, y_axis, changed


def infer_line_axes_from_visual_structure(img, merged_lines, x_axis=None, y_axis=None):
    """Synthesize line-chart axes from gridlines or the plotted line when axes are weak."""
    if img is None:
        return x_axis, y_axis, False
    if x_axis is not None and y_axis is not None:
        return x_axis, y_axis, False

    h, w = img.shape[:2]
    horizontals = _horizontal_gridline_candidates(merged_lines, img.shape)
    if len(horizontals) >= 2:
        ys = sorted(_dedupe_pixels([item[2] for item in horizontals]))
        relevant = [item for item in horizontals if item[2] in set(ys)]
        left = int(np.percentile([item[0] for item in relevant], 20))
        right = int(np.percentile([item[1] for item in relevant], 80))
        top = int(min(ys))
        bottom = int(max(ys))
        if right - left > w * 0.25 and bottom - top > h * 0.18:
            return [left, bottom, right, bottom], [left, bottom, left, top], True

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Saturated plotted lines/markers are more reliable than text for weak-axis line charts.
    mask = (hsv[:, :, 1] > 45) & (hsv[:, :, 2] > 45)
    mask[:, : int(w * 0.02)] = False
    mask[: int(h * 0.02), :] = False
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < max(8, w * h * 0.00002):
            continue
        if bw < 3 and bh < 3:
            continue
        boxes.append((x, y, bw, bh))
    if not boxes:
        return x_axis, y_axis, False

    left = min(x for x, _, _, _ in boxes)
    right = max(x + bw for x, _, bw, _ in boxes)
    top = min(y for _, y, _, _ in boxes)
    bottom = max(y + bh for _, y, _, bh in boxes)
    if right - left < w * 0.18 or bottom - top < h * 0.12:
        return x_axis, y_axis, False

    pad_x = max(4, int((right - left) * 0.01))
    pad_y = max(4, int((bottom - top) * 0.02))
    left = max(0, left - pad_x)
    right = min(w - 1, right + pad_x)
    top = max(0, top - pad_y)
    bottom = min(h - 1, bottom + pad_y)
    return [left, bottom, right, bottom], [left, bottom, left, top], True


def _regular_grid_pixels(values, *, min_count=4):
    pixels = sorted(_dedupe_pixels(values))
    if len(pixels) < min_count:
        return []

    gaps = [pixels[index + 1] - pixels[index] for index in range(len(pixels) - 1)]
    usable_gaps = [gap for gap in gaps if gap >= 8]
    if not usable_gaps:
        return []
    median_gap = float(np.median(usable_gaps))
    if median_gap <= 0:
        return []

    groups = [[pixels[0]]]
    for previous, current in zip(pixels, pixels[1:]):
        gap = current - previous
        if median_gap * 0.45 <= gap <= median_gap * 2.25:
            groups[-1].append(current)
        else:
            groups.append([current])

    groups = [group for group in groups if len(group) >= min_count]
    if not groups:
        return []
    return max(groups, key=len)


def refine_point_chart_axes_from_gridlines(lines, x_axis, y_axis, chart_type, image_shape):
    """Recover scatter/bubble plot bounds when an internal gridline is mistaken for an axis.

    Some real-world bubble charts render faint axes and stronger internal grid
    lines. The generic axis scorer can then choose an internal vertical gridline
    as the y-axis. This correction is intentionally conservative: it only runs
    for point charts with a regular grid and only changes axes when the inferred
    plot boundary is substantially different from the current axis.
    """
    chart_type = (chart_type or "").lower()
    if chart_type not in {"scatter", "bubble"} or x_axis is None or y_axis is None:
        return x_axis, y_axis, False

    h, w = image_shape[:2]
    verticals = []
    for line in lines or []:
        x1, y1, x2, y2 = [int(value) for value in line]
        if abs(x1 - x2) > 4:
            continue
        top, bottom = sorted([y1, y2])
        length = bottom - top
        if length < h * 0.45:
            continue
        if top > h * 0.35 or bottom < h * 0.55:
            continue
        x = int(round((x1 + x2) / 2))
        verticals.append((x, top, bottom))

    x_grid = _regular_grid_pixels([item[0] for item in verticals], min_count=4)
    if len(x_grid) < 4:
        return x_axis, y_axis, False

    grid_left, grid_right = min(x_grid), max(x_grid)
    grid_width = grid_right - grid_left
    if grid_width < w * 0.25:
        return x_axis, y_axis, False

    relevant_verticals = [item for item in verticals if grid_left - 3 <= item[0] <= grid_right + 3]
    vertical_top = int(np.percentile([item[1] for item in relevant_verticals], 25))
    vertical_bottom = int(np.percentile([item[2] for item in relevant_verticals], 50))

    horizontals = []
    for line in lines or []:
        x1, y1, x2, y2 = [int(value) for value in line]
        if abs(y1 - y2) > 4:
            continue
        left, right = sorted([x1, x2])
        y = int(round((y1 + y2) / 2))
        if y < vertical_top - 8 or y > vertical_bottom + max(12, int(h * 0.05)):
            continue
        overlap = max(0, min(right, grid_right) - max(left, grid_left))
        if overlap < grid_width * 0.55:
            continue
        horizontals.append((left, right, y))

    y_grid = _regular_grid_pixels([item[2] for item in horizontals], min_count=4)
    if len(y_grid) < 4:
        return x_axis, y_axis, False

    candidate_top = min(y_grid)
    candidate_bottom = max(y_grid)
    candidate_left = grid_left
    candidate_right = int(np.median([item[1] for item in horizontals if item[2] in set(y_grid)]))

    current_left = _axis_x(y_axis)
    current_bottom = _axis_y(x_axis)
    left_shift = abs(current_left - candidate_left)
    bottom_shift = abs(current_bottom - candidate_bottom)
    if left_shift < max(18, grid_width * 0.06) and bottom_shift < max(10, h * 0.025):
        return x_axis, y_axis, False

    if candidate_right <= candidate_left or candidate_bottom <= candidate_top:
        return x_axis, y_axis, False

    repaired_x_axis = [candidate_left, candidate_bottom, candidate_right, candidate_bottom]
    repaired_y_axis = [candidate_left, candidate_bottom, candidate_left, candidate_top]
    return repaired_x_axis, repaired_y_axis, True


def _horizontal_gridline_candidates(lines, image_shape):
    h, w = image_shape[:2]
    candidates = []
    for line in lines or []:
        x1, y1, x2, y2 = [int(value) for value in line]
        if abs(y1 - y2) > 4:
            continue
        left, right = sorted([x1, x2])
        length = right - left
        if length < max(50, w * 0.28):
            continue
        y = int(round((y1 + y2) / 2))
        if y < h * 0.04 or y > h * 0.96:
            continue
        candidates.append((left, right, y, length))
    return candidates


def refine_vbar_positive_axes_from_plot_bounds(
    lines,
    x_axis,
    y_axis,
    boxes,
    y_tick_values,
    image_shape,
    axis_repair_hint=None,
):
    """Move a mistaken internal x-axis down to the positive bar baseline."""
    if x_axis is None or y_axis is None or not boxes:
        return x_axis, y_axis, False
    numeric = _finite_numeric_sequence(y_tick_values)
    if not numeric or min(numeric) < 0:
        return x_axis, y_axis, False

    h, w = image_shape[:2]
    box_left = min(box[0] for box in boxes)
    box_right = max(box[0] + box[2] for box in boxes)
    box_top = min(box[1] for box in boxes)
    box_bottom = max(box[1] + box[3] for box in boxes)
    current_axis_y = _axis_y(x_axis)

    if current_axis_y >= box_bottom - max(8, h * 0.025):
        return x_axis, y_axis, False

    candidates = _horizontal_gridline_candidates(lines, image_shape)
    relevant = []
    for left, right, y, length in candidates:
        overlap = max(0, min(right, box_right) - max(left, box_left))
        if overlap >= max(20, (box_right - box_left) * 0.35):
            relevant.append((left, right, y, length))

    if relevant:
        plot_left = int(round(float(np.percentile([item[0] for item in relevant], 10))))
        plot_right = int(round(float(np.percentile([item[1] for item in relevant], 90))))
        grid_top = min(item[2] for item in relevant)
        grid_bottom = max(item[2] for item in relevant)
    else:
        plot_left = box_left
        plot_right = box_right
        grid_top = box_top
        grid_bottom = box_bottom

    plot_bottom = max(grid_bottom, box_bottom)
    if plot_bottom - current_axis_y < max(18, h * 0.08):
        return x_axis, y_axis, False

    margin = max(2, int(round(w * 0.005)))
    plot_left = max(0, min(plot_left, box_left) - margin)
    plot_right = min(w - 1, max(plot_right, box_right) + margin)
    plot_top = max(0, min(grid_top, box_top))
    if plot_right <= plot_left or plot_bottom <= plot_top:
        return x_axis, y_axis, False

    repaired_x_axis, repaired_y_axis = _vbar_axes_from_bounds(
        plot_left,
        plot_right,
        plot_top,
        plot_bottom,
        axis_repair_hint,
    )
    return repaired_x_axis, repaired_y_axis, True


def refine_vbar_axes_from_grid_bounds(lines, x_axis, y_axis, boxes, image_shape, axis_repair_hint=None):
    """Recover plot bounds when an outer frame or inner gridline is mistaken for an axis."""
    if x_axis is None or y_axis is None:
        return x_axis, y_axis, False

    h, w = image_shape[:2]
    candidates = _horizontal_gridline_candidates(lines, image_shape)
    if len(candidates) < 3:
        return x_axis, y_axis, False

    if boxes:
        box_left = min(box[0] for box in boxes)
        box_right = max(box[0] + box[2] for box in boxes)
        relevant = []
        for left, right, y, length in candidates:
            overlap = max(0, min(right, box_right) - max(left, box_left))
            if overlap >= max(20, (box_right - box_left) * 0.25):
                relevant.append((left, right, y, length))
        if len(relevant) >= 3:
            candidates = relevant

    plot_left = int(round(float(np.percentile([item[0] for item in candidates], 10))))
    plot_right = int(round(float(np.percentile([item[1] for item in candidates], 90))))
    plot_top = min(item[2] for item in candidates)
    plot_bottom = max(item[2] for item in candidates)
    if plot_right <= plot_left or plot_bottom <= plot_top:
        return x_axis, y_axis, False

    current_axis_y = _axis_y(x_axis)
    current_axis_x = _axis_x(y_axis)
    x_axis_span = abs(int(x_axis[2]) - int(x_axis[0]))
    plot_span = plot_right - plot_left

    axis_at_image_border = (
        current_axis_y > h * 0.94
        or current_axis_x < w * 0.03
        or current_axis_x > w * 0.97
    )
    axis_inside_plot = current_axis_y < plot_bottom - max(16, h * 0.04)
    axis_not_covering_plot = x_axis_span < plot_span * 0.65
    axis_left_far_from_grid = abs(current_axis_x - plot_left) > max(20, w * 0.045)

    if not (axis_at_image_border or axis_inside_plot or axis_not_covering_plot or axis_left_far_from_grid):
        return x_axis, y_axis, False

    repaired_x_axis, repaired_y_axis = _vbar_axes_from_bounds(
        plot_left,
        plot_right,
        plot_top,
        plot_bottom,
        axis_repair_hint,
    )
    return repaired_x_axis, repaired_y_axis, True


def _linear_numeric_tick_pixels_from_bounds(tick_values, plot_top, plot_bottom):
    numeric = _finite_numeric_sequence(tick_values)
    if not numeric or len(numeric) < 2:
        return []
    value_min = min(numeric)
    value_max = max(numeric)
    if value_max == value_min:
        return []
    height = float(plot_bottom - plot_top)
    pixels = [
        int(round(plot_bottom - ((value - value_min) / (value_max - value_min)) * height))
        for value in numeric
    ]
    return pixels


def _dark_vertical_axis_near_bars(img, box_left, plot_top, plot_bottom):
    if img is None or plot_bottom <= plot_top:
        return None
    h, w = img.shape[:2]
    search_left = max(0, int(round(w * 0.035)))
    search_right = min(w - 1, int(round(box_left + max(50, w * 0.10))))
    if search_right <= search_left:
        return None
    y1 = max(0, int(plot_top))
    y2 = min(h - 1, int(plot_bottom))
    if y2 - y1 < max(40, int(h * 0.08)):
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2 + 1, search_left:search_right + 1]
    dark = roi < 90
    counts = dark.sum(axis=0)
    if len(counts) >= 5:
        counts = np.convolve(counts, np.ones(3), mode="same")
    min_count = max(40, int((y2 - y1 + 1) * 0.38))
    candidate_indices = np.where(counts >= min_count)[0]
    if len(candidate_indices) == 0:
        return None

    groups = []
    current = [int(candidate_indices[0])]
    for index in candidate_indices[1:]:
        index = int(index)
        if index - current[-1] <= 3:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
    groups.append(current)
    best_group = max(groups, key=lambda group: float(np.max(counts[group])))
    center = int(round(float(np.median(best_group)))) + search_left
    return center


def _regular_numeric_ticks_have_irregular_pixels(tick_values, pixel_positions):
    numeric = _finite_numeric_sequence(tick_values)
    if not numeric or len(numeric) < 4 or len(pixel_positions or []) != len(numeric):
        return False
    value_gaps = np.abs(np.diff(np.array(numeric, dtype=float)))
    pixel_gaps = np.abs(np.diff(np.array(pixel_positions, dtype=float)))
    if np.any(value_gaps <= 0) or np.any(pixel_gaps <= 0):
        return True
    value_cv = float(np.std(value_gaps) / max(1e-9, np.mean(value_gaps)))
    if value_cv > 0.12:
        return False
    median_gap = float(np.median(pixel_gaps))
    if median_gap <= 0:
        return True
    pixel_cv = float(np.std(pixel_gaps) / max(1e-9, np.mean(pixel_gaps)))
    return pixel_cv > 0.35 or float(np.min(pixel_gaps)) < median_gap * 0.45


def _saturated_point_bbox(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Exclude bottom legends/color bars while keeping low plot points.
    y_limit = int(round(h * 0.86))
    mask = (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 60)
    mask[y_limit:h, :] = False
    mask[:, 0:int(round(w * 0.03))] = False
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < 10 or area > max(900, w * h * 0.004):
            continue
        if bw > 45 or bh > 45:
            continue
        boxes.append((x, y, bw, bh))
    if len(boxes) < 6:
        return None
    left = min(x for x, y, bw, bh in boxes)
    right = max(x + bw for x, y, bw, bh in boxes)
    top = min(y for x, y, bw, bh in boxes)
    bottom = max(y + bh for x, y, bw, bh in boxes)
    if right - left < w * 0.45 or bottom - top < h * 0.35:
        return None
    return left, top, right, bottom


def _point_plot_bounds_from_grid_or_points(img, x_axis=None, y_axis=None):
    """Return conservative scatter/bubble plot bounds from grid first, points second."""
    if img is None:
        return None
    h, w = img.shape[:2]
    gray_bounds = infer_point_plot_bounds_from_gray_horizontal_grid(img)
    if gray_bounds:
        left, right, top, bottom, _ = gray_bounds
        return int(left), int(top), int(right), int(bottom)

    bbox = _saturated_point_bbox(img)
    if x_axis is not None and y_axis is not None:
        left = int(_axis_x(y_axis))
        right = int(max(x_axis[0], x_axis[2]))
        bottom = int(_axis_y(x_axis))
        top = int(min(y_axis[1], y_axis[3]))
        if bbox:
            box_left, box_top, box_right, box_bottom = bbox
            pad_x = max(4, int(round((box_right - box_left) * 0.025)))
            pad_y = max(4, int(round((box_bottom - box_top) * 0.035)))
            left = min(left, max(0, int(box_left) - pad_x))
            right = max(right, min(w - 1, int(box_right) + pad_x))
            top = min(top, max(int(round(h * 0.04)), int(box_top) - pad_y))
            bottom = max(bottom, min(h - 1, int(box_bottom) + pad_y))
        if right - left >= w * 0.25 and bottom - top >= h * 0.18:
            return left, top, right, bottom

    if not bbox:
        return None
    left, top, right, bottom = bbox
    pad_x = max(8, int(round((right - left) * 0.06)))
    pad_y = max(8, int(round((bottom - top) * 0.08)))
    left = max(0, int(left) - pad_x)
    right = min(w - 1, int(right) + pad_x)
    top = max(int(round(h * 0.04)), int(top) - pad_y)
    bottom = min(h - 1, int(bottom) + pad_y)
    if right - left < w * 0.25 or bottom - top < h * 0.18:
        return None
    return left, top, right, bottom


def _numeric_pixels_on_bounds(values, direction, start, end, scale="linear"):
    numeric = _finite_numeric_sequence(values)
    if not numeric or len(numeric) < 2:
        return []
    arr = np.array(numeric, dtype=float)
    if str(scale or "linear").lower() == "log":
        if np.any(arr <= 0):
            return []
        arr = np.log10(arr)
    value_min = float(np.min(arr))
    value_max = float(np.max(arr))
    if not math.isfinite(value_min) or not math.isfinite(value_max) or value_max == value_min:
        return []
    ratios = (arr - value_min) / (value_max - value_min)
    if direction == "y":
        pixels = [int(round(float(start) - ratio * (float(start) - float(end)))) for ratio in ratios]
    else:
        pixels = [int(round(float(start) + ratio * (float(end) - float(start)))) for ratio in ratios]
    return pixels


def repair_point_tick_span_from_plot_bounds(
    img,
    chart_type,
    x_axis,
    y_axis,
    x_tick_values,
    y_tick_values,
    x_pixel_positions,
    y_pixel_positions,
    x_axis_scale,
    y_axis_scale,
    axis_repair_hint,
):
    # 如果点图 tick 只覆盖了局部网格 patch，这里用绘图区边界把 tick 映射扩展回全图范围。
    """Expand scatter/bubble tick binding when it only covers a local grid patch."""
    if img is None or (chart_type or "").lower() not in {"scatter", "bubble"}:
        return x_axis, y_axis, x_pixel_positions, y_pixel_positions, x_axis_scale, y_axis_scale, False
    hint = normalize_axis_repair_hint(axis_repair_hint)
    if not (hint.get("has_background_grid") or _point_chart_grid_hint(hint) or axis_repair_enabled(hint)):
        return x_axis, y_axis, x_pixel_positions, y_pixel_positions, x_axis_scale, y_axis_scale, False

    bounds = _point_plot_bounds_from_grid_or_points(img, x_axis, y_axis)
    if not bounds:
        return x_axis, y_axis, x_pixel_positions, y_pixel_positions, x_axis_scale, y_axis_scale, False
    left, top, right, bottom = bounds
    width = max(1, right - left)
    height = max(1, bottom - top)
    changed = False

    def span(values):
        return max(values) - min(values) if len(values or []) >= 2 else 0

    if (
        normalize_axis_type_name("numeric") == NUMERIC_AXIS_TYPE
        and len(x_tick_values or []) >= 2
        and span(x_pixel_positions) < width * 0.68
    ):
        repaired = _numeric_pixels_on_bounds(x_tick_values, "x", left, right, x_axis_scale)
        if len(repaired) == len(x_tick_values or []):
            x_pixel_positions = repaired
            changed = True

    if (
        len(y_tick_values or []) >= 2
        and span(y_pixel_positions) < height * 0.68
    ):
        repaired = _numeric_pixels_on_bounds(y_tick_values, "y", bottom, top, y_axis_scale)
        if len(repaired) == len(y_tick_values or []):
            y_pixel_positions = repaired
            changed = True

    if not changed:
        return x_axis, y_axis, x_pixel_positions, y_pixel_positions, x_axis_scale, y_axis_scale, False

    return (
        [left, bottom, right, bottom],
        [left, bottom, left, top],
        x_pixel_positions,
        y_pixel_positions,
        axis_scale_from_ticks_and_pixels(x_tick_values, x_pixel_positions),
        axis_scale_from_ticks_and_pixels(y_tick_values, y_pixel_positions),
        True,
    )


def _normalized_prefix_ticks(values):
    numeric = _finite_numeric_sequence(values)
    if not numeric or len(numeric) < 4:
        return None
    gaps = np.diff(np.array(numeric, dtype=float))
    if np.any(gaps <= 0):
        return None
    step = float(np.median(gaps))
    if step <= 0 or step > 0.25:
        return None
    if abs(float(numeric[0])) > max(1e-6, step * 0.2):
        return None
    if max(numeric) > 1.05:
        return None
    if float(np.std(gaps)) / max(1e-9, abs(step)) > 0.08:
        return None
    return numeric, step


def repair_normalized_point_prefix_axis(
    img,
    chart_type,
    x_axis,
    y_axis,
    x_tick_values,
    y_tick_values,
    x_pixel_positions,
    y_pixel_positions,
    axis_repair_hint,
):
    # 处理 0~1 归一化轴只识别到前缀的情况。
    # 如果点云和网格表明坐标应覆盖完整 0~1，则补齐 tick 和轴范围。
    if chart_type not in {"scatter", "bubble"}:
        return x_axis, y_axis, x_tick_values, y_tick_values, x_pixel_positions, y_pixel_positions, False
    hint = normalize_axis_repair_hint(axis_repair_hint)
    if not hint.get("has_background_grid") and hint.get("plot_area_style") != "grid_only":
        return x_axis, y_axis, x_tick_values, y_tick_values, x_pixel_positions, y_pixel_positions, False
    x_info = _normalized_prefix_ticks(x_tick_values)
    y_info = _normalized_prefix_ticks(y_tick_values)
    if not x_info or not y_info:
        return x_axis, y_axis, x_tick_values, y_tick_values, x_pixel_positions, y_pixel_positions, False
    bbox = _saturated_point_bbox(img)
    if not bbox:
        return x_axis, y_axis, x_tick_values, y_tick_values, x_pixel_positions, y_pixel_positions, False
    left, top, right, bottom = bbox
    h, w = img.shape[:2]
    margin_x = max(2, int(round(w * 0.004)))
    margin_y = max(2, int(round(h * 0.004)))
    left = max(0, left - margin_x)
    right = min(w - 1, right + margin_x)
    top = max(0, top - margin_y)
    bottom = min(h - 1, bottom + margin_y)

    current_right = max(x_pixel_positions or [0])
    current_bottom = max(y_pixel_positions or [0])
    x_numeric, x_step = x_info
    y_numeric, y_step = y_info
    x_full = max(x_numeric) >= 0.8
    y_full = max(y_numeric) >= 0.8
    if (
        x_full
        and y_full
        and len(x_pixel_positions or []) >= len(x_numeric)
        and len(y_pixel_positions or []) >= len(y_numeric)
        and current_right >= right - max(20, int(w * 0.04))
        and current_bottom <= bottom + max(20, int(h * 0.04))
    ):
        return x_axis, y_axis, x_tick_values, y_tick_values, x_pixel_positions, y_pixel_positions, False

    x_end = 1.0 if x_full or max(x_numeric) <= 0.55 else max(x_numeric)
    y_end = 1.0 if y_full or max(y_numeric) <= 0.55 else max(y_numeric)
    x_count = int(round(x_end / x_step)) + 1
    y_count = int(round(y_end / y_step)) + 1
    if not (5 <= x_count <= 21 and 5 <= y_count <= 21):
        return x_axis, y_axis, x_tick_values, y_tick_values, x_pixel_positions, y_pixel_positions, False

    repaired_x_ticks = [round(index * x_step, 10) for index in range(x_count)]
    repaired_y_ticks = [round(index * y_step, 10) for index in range(y_count)]
    repaired_x_pixels = [int(round(value)) for value in np.linspace(left, right, x_count)]
    repaired_y_pixels = [int(round(value)) for value in np.linspace(bottom, top, y_count)]
    repaired_x_axis = [left, bottom, right, bottom]
    repaired_y_axis = [left, bottom, left, top]
    return (
        repaired_x_axis,
        repaired_y_axis,
        repaired_x_ticks,
        repaired_y_ticks,
        repaired_x_pixels,
        repaired_y_pixels,
        True,
    )


def repair_suspicious_vbar_value_axis(
    img,
    lines,
    x_axis,
    y_axis,
    boxes,
    y_tick_values,
    y_pixel_positions,
    axis_repair_hint,
):
    # v_bar 数值轴位置异常时的保护性修复。
    # 只在轴明显跑偏、像素跨度塌缩或右侧轴位置不符合先验时触发，
    # 避免影响原数据集中已经正确的柱形图。
    """Repair v_bar value-axis geometry only when the current y mapping is clearly collapsed."""
    numeric = _finite_numeric_sequence(y_tick_values)
    if img is None or not boxes or not numeric or len(numeric) < 2:
        return x_axis, y_axis, y_pixel_positions, "linear", False

    h, w = img.shape[:2]
    box_left = min(box[0] for box in boxes)
    box_right = max(box[0] + box[2] for box in boxes)
    box_top = min(box[1] for box in boxes)
    box_bottom = max(box[1] + box[3] for box in boxes)
    box_height = max(1, box_bottom - box_top)

    axis_x = _axis_x(y_axis) if y_axis is not None else None
    axis_span = abs(int(y_axis[1]) - int(y_axis[3])) if y_axis is not None else 0
    pixel_span = (
        max(y_pixel_positions) - min(y_pixel_positions)
        if len(y_pixel_positions or []) >= 2
        else 0
    )
    hint = normalize_axis_repair_hint(axis_repair_hint)
    expected_left_axis = hint.get("y_axis_position") in {"left", "unknown", "none"}
    expected_right_axis = hint.get("y_axis_position") == "right"
    axis_far_right = axis_x is not None and axis_x > box_right + max(24, int(w * 0.04))
    axis_far_from_left = (
        expected_left_axis
        and axis_x is not None
        and abs(axis_x - box_left) > max(40, int(w * 0.08))
    )
    axis_wrong_side_right = (
        expected_right_axis
        and axis_x is not None
        and axis_x < box_right - max(30, int(w * 0.05))
    )
    collapsed_axis = axis_span < max(45, int(box_height * 0.45))
    collapsed_pixels = pixel_span < max(45, int(box_height * 0.45))
    if not (axis_far_right or axis_far_from_left or axis_wrong_side_right or collapsed_axis or collapsed_pixels):
        return x_axis, y_axis, y_pixel_positions, "linear", False

    candidates = _horizontal_gridline_candidates(lines, img.shape)
    relevant = []
    for left, right, y, length in candidates:
        if y > h * 0.94:
            continue
        if y > box_bottom + max(30, int(box_height * 0.28)):
            continue
        overlap = max(0, min(right, box_right) - max(left, box_left))
        if overlap >= max(24, (box_right - box_left) * 0.18):
            relevant.append((left, right, y, length))

    if len(relevant) >= 2:
        plot_left = int(round(float(np.percentile([item[0] for item in relevant], 10))))
        plot_right = int(round(float(np.percentile([item[1] for item in relevant], 90))))
        plot_top = min(item[2] for item in relevant)
        plot_bottom = max(item[2] for item in relevant)
    else:
        margin = max(2, int(round(w * 0.004)))
        plot_left = max(0, box_left - margin)
        plot_right = min(w - 1, box_right + margin)
        plot_top = max(0, box_top)
        plot_bottom = min(h - 1, box_bottom)

    if x_axis is not None:
        existing_axis_y = _axis_y(x_axis)
        if (
            plot_top < existing_axis_y < h * 0.94
            and existing_axis_y <= box_bottom + max(30, int(box_height * 0.28))
        ):
            plot_bottom = max(plot_bottom, existing_axis_y)

    detected_y_axis = _vertical_axis_from_lines(lines, w, h)
    accepted_detected_axis = False
    if detected_y_axis is not None:
        detected_x = _axis_x(detected_y_axis)
        detected_top = min(int(detected_y_axis[1]), int(detected_y_axis[3]))
        detected_bottom = max(int(detected_y_axis[1]), int(detected_y_axis[3]))
        detected_span = detected_bottom - detected_top
        left_axis_candidate = (
            detected_x > w * 0.035
            and detected_x < box_left + max(60, int(w * 0.10))
            and detected_span >= max(45, int(box_height * 0.45))
        )
        right_axis_candidate = (
            expected_right_axis
            and detected_x > box_right - max(60, int(w * 0.10))
            and detected_x < w * 0.97
            and detected_span >= max(45, int(box_height * 0.45))
        )
        if left_axis_candidate or right_axis_candidate:
            if right_axis_candidate:
                plot_right = detected_x
            else:
                plot_left = detected_x
            plot_top = min(plot_top, detected_top)
            plot_bottom = max(plot_bottom, detected_bottom)
            accepted_detected_axis = True
    if not accepted_detected_axis and not expected_right_axis:
        dark_axis_x = _dark_vertical_axis_near_bars(img, box_left, plot_top, plot_bottom)
        if dark_axis_x is not None:
            plot_left = dark_axis_x

    if plot_right <= plot_left or plot_bottom <= plot_top:
        return x_axis, y_axis, y_pixel_positions, "linear", False

    repaired_x_axis, repaired_y_axis = _vbar_axes_from_bounds(
        plot_left,
        plot_right,
        plot_top,
        plot_bottom,
        axis_repair_hint,
    )
    recovered_pixels, recovered_scale = recover_bar_value_axis_pixels_from_grid(
        img,
        lines,
        repaired_x_axis,
        repaired_y_axis,
        "v_bar",
        "y",
        y_tick_values,
    )
    if len(recovered_pixels) != len(numeric):
        recovered_pixels = _linear_numeric_tick_pixels_from_bounds(
            y_tick_values,
            plot_top,
            plot_bottom,
        )
        recovered_scale = "linear"
    elif _regular_numeric_ticks_have_irregular_pixels(y_tick_values, recovered_pixels):
        linear_pixels = _linear_numeric_tick_pixels_from_bounds(
            y_tick_values,
            plot_top,
            plot_bottom,
        )
        if len(linear_pixels) == len(numeric):
            recovered_pixels = linear_pixels
            recovered_scale = "linear"

    if len(recovered_pixels) != len(numeric):
        return x_axis, y_axis, y_pixel_positions, "linear", False

    axis_bottom = max(int(pixel) for pixel in recovered_pixels)
    axis_top = min(int(pixel) for pixel in recovered_pixels)
    repaired_x_axis, repaired_y_axis = _vbar_axes_from_bounds(
        plot_left,
        plot_right,
        axis_top,
        axis_bottom,
        axis_repair_hint,
    )

    return repaired_x_axis, repaired_y_axis, recovered_pixels, recovered_scale, True


def apply_bar_geometry_repair_hint(
    chart_type,
    axis_repair_hint,
    boxes,
    x_tick_count=0,
    y_tick_count=0,
):
    # 根据柱体几何和当前 tick 数量，对上传阶段的 axis_repair_hint 做二次补充。
    # 例如：柱体明显存在但类别轴 tick 不足时，允许后续从柱体中心合成类别 tick。
    """Enable repair only for obvious bar/tick mismatches.

    The MLLM hint stays the primary switch. This guard catches cases where the
    model says axes are present but CV finds far fewer category ticks than bars.
    """
    hint = normalize_axis_repair_hint(axis_repair_hint)
    chart_type = (chart_type or "").lower()
    orientation = _bar_orientation(chart_type)
    if not orientation or len(boxes or []) < 1:
        return hint

    min_expected_ticks = max(1, int(np.ceil(len(boxes) * 0.6)))
    if orientation == "h":
        if y_tick_count < min_expected_ticks:
            hint["y_ticks_missing"] = True
            hint["reason"] = (
                hint.get("reason") or ""
            ) + f" | geometry repair: {y_tick_count} y ticks for {len(boxes)} bars"
        if x_tick_count < 2:
            hint["x_axis_missing"] = True
            hint["x_ticks_missing"] = True
            hint["reason"] = (
                hint.get("reason") or ""
            ) + f" | geometry repair: {x_tick_count} numeric x ticks"
    elif orientation == "v":
        if x_tick_count < min_expected_ticks:
            hint["x_ticks_missing"] = True
            hint["reason"] = (
                hint.get("reason") or ""
            ) + f" | geometry repair: {x_tick_count} x ticks for {len(boxes)} bars"
        if y_tick_count < 2:
            hint["y_axis_missing"] = True
            hint["y_ticks_missing"] = True
            hint["reason"] = (
                hint.get("reason") or ""
            ) + f" | geometry repair: {y_tick_count} numeric y ticks"
    return hint


def _numeric_sequence(values):
    numeric = []
    for value in values or []:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return None
    return numeric


def _numeric_ticks_from_unit_labels(values):
    numeric = []
    for value in values or []:
        if isinstance(value, (int, float)):
            numeric.append(float(value))
            continue
        text = str(value).replace(",", "").strip()
        match = __import__("re").search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            numeric.append(float(match.group(0)))
        except ValueError:
            return None
    return numeric if len(numeric) == len(values or []) else None


def _strict_numeric_progression(values):
    numeric = _numeric_ticks_from_unit_labels(values)
    if numeric is None or len(numeric) < 2:
        return None
    diffs = np.diff(np.array(numeric, dtype=float))
    if np.any(np.abs(diffs) <= 1e-12):
        return None
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        return None
    return numeric


def _relative_range_mismatch(a_values, b_values):
    a_numeric = _numeric_ticks_from_unit_labels(a_values)
    b_numeric = _numeric_ticks_from_unit_labels(b_values)
    if not a_numeric or not b_numeric or len(a_numeric) < 2 or len(b_numeric) < 2:
        return False
    a_range = max(a_numeric) - min(a_numeric)
    b_range = max(b_numeric) - min(b_numeric)
    if a_range <= 0 or b_range <= 0:
        return False
    ratio = max(a_range, b_range) / max(1e-9, min(a_range, b_range))
    return ratio >= 5.0


# 上传阶段 MLLM 读到的 tick 只作为“有条件的先验”。
# 只有当前 tick 文本无法解析、数量与像素明显不匹配或数值范围异常时，才使用它覆盖。
def _apply_upload_numeric_axis_prior(
    chart_type,
    direction,
    current_values,
    axis_type,
    pixel_positions,
    axis_repair_hint,
):
    """Use upload-time MLLM tick labels only as a guarded numeric-axis prior."""
    hint = normalize_axis_repair_hint(axis_repair_hint)
    prior = hint.get("axis_tick_labels") if isinstance(hint.get("axis_tick_labels"), dict) else {}
    prior_values = prior.get(f"{direction}_ticks") or []
    prior_axis_type = normalize_axis_type_name(prior.get(f"{direction}_axis_type", "unknown"))
    role = hint.get(f"{direction}_axis_role")
    is_numeric_role = (
        prior_axis_type == NUMERIC_AXIS_TYPE
        or prior_axis_type == "numeric"
        or role == "numeric"
        or (
            chart_type in {"scatter", "bubble"}
            and direction in {"x", "y"}
        )
        or (
            chart_type == "line"
            and direction == "y"
        )
        or (
            chart_type in {"v_bar", "v_stacked_bar"}
            and direction == "y"
        )
        or (
            chart_type in {"h_bar", "h_stacked_bar"}
            and direction == "x"
        )
    )
    if not is_numeric_role:
        return current_values, axis_type, False

    prior_numeric = _strict_numeric_progression(prior_values)
    if not prior_numeric:
        return current_values, axis_type, False
    if len(prior_numeric) > 30:
        return current_values, axis_type, False

    current_numeric = _numeric_ticks_from_unit_labels(current_values)
    pixel_count = len(pixel_positions or [])
    current_bad = (
        current_numeric is None
        or len(current_numeric) < 2
        or _strict_numeric_progression(current_values) is None
    )
    if not current_bad and pixel_count >= 2:
        current_gap = abs(len(current_numeric) - pixel_count)
        prior_gap = abs(len(prior_numeric) - pixel_count)
        if prior_gap + 1 < current_gap:
            current_bad = True
    if not current_bad and _relative_range_mismatch(current_values, prior_values):
        current_bad = True
    if not current_bad:
        return current_values, axis_type, False

    return [float(value) for value in prior_numeric], NUMERIC_AXIS_TYPE, True


def coerce_chart_axis_numeric_ticks(chart_type, axis, tick_values, axis_type):
    """Value axes often show numeric ticks with units, e.g. '$25' or '65 gr'."""
    chart_type = (chart_type or "").lower()
    axis = (axis or "").lower()
    value_axis = (
        chart_type in {"scatter", "bubble"}
        or ((chart_type in {"v_bar", "v_stacked_bar", "line"}) and axis == "y")
        or (chart_type in {"h_bar", "h_stacked_bar"} and axis == "x")
    )
    if not value_axis:
        return tick_values, axis_type
    numeric = _numeric_ticks_from_unit_labels(tick_values)
    if numeric is None or len(numeric) < 2:
        return tick_values, axis_type
    return numeric, NUMERIC_AXIS_TYPE


def normalize_axis_type_name(axis_type):
    text = str(axis_type or "").strip().lower()
    if text in {
        "numeric",
        "number",
        "value",
        "quantitative",
        NUMERIC_AXIS_TYPE.lower(),
        "鏁板€艰酱",
        "鏁板€艰酱".lower(),
    }:
        return NUMERIC_AXIS_TYPE
    if text in {
        "text",
        "category",
        "categorical",
        "date",
        "time",
        TEXT_AXIS_TYPE.lower(),
        "鏂囧瓧杞?",
        "鏂囧瓧杞?".lower(),
    }:
        return TEXT_AXIS_TYPE
    return axis_type


def _linspace_pixels_on_axis(direction, axis, count):
    if axis is None or count <= 0:
        return []
    if direction == "x":
        start, end = sorted([int(axis[0]), int(axis[2])])
    else:
        low, high = sorted([int(axis[1]), int(axis[3])])
        start, end = high, low
    return [int(round(value)) for value in np.linspace(start, end, count)]


def bind_noisy_numeric_bar_ticks_to_labels(
    chart_type,
    direction,
    axis,
    pixel_positions,
    tick_values,
):
    # bar 数值轴候选 tick 过多时，按 MLLM 读到的标签数量重新绑定，减少网格线/边框误检。
    """Limit value-axis tick candidates to the count read by the MLLM."""
    if not _is_bar_chart_type(chart_type) or not tick_values or len(tick_values) < 2:
        return pixel_positions, False
    if direction == "x" and chart_type not in {"h_bar", "h_stacked_bar"}:
        return pixel_positions, False
    if direction == "y" and chart_type not in {"v_bar", "v_stacked_bar"}:
        return pixel_positions, False
    if len(pixel_positions or []) <= max(len(tick_values) * 2, len(tick_values) + 3):
        return pixel_positions, False
    repaired = _linspace_pixels_on_axis(direction, axis, len(tick_values))
    return (repaired, True) if repaired else (pixel_positions, False)


def bind_noisy_numeric_ticks_to_labels(
    img,
    merged_lines,
    x_axis,
    y_axis,
    chart_type,
    direction,
    axis,
    pixel_positions,
    tick_values,
):
    # line/scatter/bubble 的数值轴候选过多或过少时，使用标签数量、轴端点和背景网格重新绑定。
    """Limit noisy non-bar numeric tick candidates to the labels read by MLLM."""
    chart_type = (chart_type or "").lower()
    if _is_bar_chart_type(chart_type) or chart_type not in {"line", "scatter", "bubble"}:
        return pixel_positions, "linear", False
    if axis is None or len(tick_values or []) < 2:
        return pixel_positions, "linear", False

    target_count = len(tick_values)
    pixels = sorted(_dedupe_pixels(pixel_positions or []), reverse=(direction == "y"))
    if len(pixels) < target_count <= 40:
        endpoint_pixels = add_missing_numeric_axis_endpoints(
            direction,
            axis,
            pixels,
            tick_values,
        )
        if len(endpoint_pixels) == target_count:
            return endpoint_pixels, axis_scale_from_ticks_and_pixels(tick_values, endpoint_pixels), True
        projected = infer_tick_pixels_from_gridlines(
            merged_lines,
            x_axis,
            y_axis,
            direction,
            img.shape,
        )
        selected, scale = select_projected_tick_pixels_for_values(
            projected,
            tick_values,
            direction,
            axis,
        )
        if len(selected) == target_count:
            return selected, scale, True
        repaired = _linspace_pixels_on_axis(direction, axis, target_count)
        if repaired:
            return repaired, axis_scale_from_ticks_and_pixels(tick_values, repaired), True
        return pixel_positions, "linear", False

    if len(pixels) <= max(target_count * 2, target_count + 3):
        return pixel_positions, "linear", False

    projected = infer_tick_pixels_from_gridlines(
        merged_lines,
        x_axis,
        y_axis,
        direction,
        img.shape,
    )
    if len(projected) < target_count:
        projected = infer_point_chart_grid_pixels_by_projection(
            img,
            x_axis,
            y_axis,
            direction,
            expected_count=target_count,
        )

    selected, scale = select_projected_tick_pixels_for_values(
        projected,
        tick_values,
        direction,
        axis,
    )
    if len(selected) == target_count:
        return selected, scale, True

    repaired = _linspace_pixels_on_axis(direction, axis, target_count)
    if repaired:
        return repaired, axis_scale_from_ticks_and_pixels(tick_values, repaired), True
    return pixel_positions, "linear", False


def _nearest_unique_pixels(pixels, candidates, direction, tolerance):
    if len(pixels or []) < 2 or len(candidates or []) < 2:
        return [], False
    ordered_pixels = [int(round(value)) for value in pixels]
    ordered_candidates = sorted(_dedupe_pixels(candidates), reverse=(direction == "y"))
    used = set()
    snapped = []
    deltas = []
    for pixel in ordered_pixels:
        available = [candidate for candidate in ordered_candidates if candidate not in used]
        if not available:
            return [], False
        nearest = min(available, key=lambda candidate: abs(int(candidate) - pixel))
        delta = abs(int(nearest) - pixel)
        if delta > tolerance:
            return [], False
        used.add(nearest)
        snapped.append(int(nearest))
        deltas.append(delta)
    if len(set(snapped)) != len(snapped):
        return [], False
    if direction == "y":
        if any(snapped[index] <= snapped[index + 1] for index in range(len(snapped) - 1)):
            return [], False
    else:
        if any(snapped[index] >= snapped[index + 1] for index in range(len(snapped) - 1)):
            return [], False
    changed = any(delta > 0 for delta in deltas)
    return snapped, changed


def snap_numeric_ticks_to_visual_grid(
    img,
    merged_lines,
    x_axis,
    y_axis,
    chart_type,
    direction,
    pixel_positions,
    tick_values,
    axis_type,
):
    # 对已经基本正确但有轻微偏移的数值 tick，吸附到附近真实网格线。
    # 容差很小，用来修正普遍的几像素偏移，而不是大幅重排正常图表。
    """Correct small numeric tick offsets by snapping to nearby visual grid lines."""
    if img is None or normalize_axis_type_name(axis_type) != NUMERIC_AXIS_TYPE:
        return pixel_positions, False
    if len(pixel_positions or []) < 2 or len(tick_values or []) < 2:
        return pixel_positions, False
    if _finite_numeric_sequence(tick_values) is None:
        return pixel_positions, False

    axis = x_axis if direction == "x" else y_axis
    if axis is None:
        return pixel_positions, False

    span = abs(int(axis[2]) - int(axis[0])) if direction == "x" else abs(int(axis[1]) - int(axis[3]))
    if span <= 0:
        return pixel_positions, False
    gaps = np.abs(np.diff(np.array(sorted([int(p) for p in pixel_positions]), dtype=float)))
    gaps = [gap for gap in gaps if gap > 0]
    median_gap = float(np.median(gaps)) if gaps else 0.0
    tolerance = int(round(max(3.0, min(10.0, span * 0.018, median_gap * 0.22 if median_gap else 10.0))))

    candidates = infer_tick_pixels_from_gridlines(
        merged_lines,
        x_axis,
        y_axis,
        direction,
        img.shape,
    )
    if len(candidates) < len(pixel_positions or []):
        projected = infer_point_chart_grid_pixels_by_projection(
            img,
            x_axis,
            y_axis,
            direction,
            expected_count=len(tick_values or []),
        )
        candidates = sorted(_dedupe_pixels(list(candidates) + list(projected)), reverse=(direction == "y"))

    snapped, changed = _nearest_unique_pixels(pixel_positions, candidates, direction, tolerance)
    if changed and len(snapped) == len(pixel_positions or []):
        return snapped, True
    return pixel_positions, False


def add_missing_numeric_axis_endpoints(direction, axis, pixel_positions, tick_values):
    """Add missing numeric endpoint pixels when tick labels include them."""
    if axis is None or len(pixel_positions) < 2 or len(tick_values or []) <= len(pixel_positions):
        return pixel_positions

    numeric = _numeric_sequence(tick_values)
    if not numeric or len(numeric) < 2:
        return pixel_positions

    pixels = list(pixel_positions)
    gaps = np.diff(sorted(pixels))
    if len(gaps) == 0:
        return pixels
    median_gap = float(np.median(gaps))
    tolerance = max(6.0, median_gap * 0.3)

    if direction == "x":
        start, end = sorted([int(axis[0]), int(axis[2])])
        first_gap = pixels[0] - start
        if len(numeric) > len(pixels) and first_gap > 0 and abs(first_gap - median_gap) <= tolerance:
            pixels.insert(0, start)
        last_gap = end - pixels[-1]
        if len(numeric) > len(pixels) and last_gap > 0 and abs(last_gap - median_gap) <= tolerance:
            pixels.append(end)
    else:
        low, high = sorted([int(axis[1]), int(axis[3])])
        first_gap = high - pixels[0]
        if len(numeric) > len(pixels) and first_gap > 0 and abs(first_gap - median_gap) <= tolerance:
            pixels.insert(0, high)
        last_gap = pixels[-1] - low
        if len(numeric) > len(pixels) and last_gap > 0 and abs(last_gap - median_gap) <= tolerance:
            pixels.append(low)

    return pixels


# ===== 第二步加密处理的网格绘制参数 =====
# 正式加密网格使用 #cccccc、1px、2px 实线 + 2px 空白循环。
# 测试脚本可通过 CHART_GRID_REVIEW_COLOR=green 临时切成绿色，便于人工检查偏移。
GRID_LINE_COLOR = (204, 204, 204)
GRID_LINE_REVIEW_COLOR = (0, 170, 0)
GRID_LINE_THICKNESS = 1
GRID_DASH_LENGTH = 2
GRID_DASH_GAP = 2


def _active_grid_line_color():
    review_color = os.getenv("CHART_GRID_REVIEW_COLOR", "").strip().lower()
    if review_color in {"green", "debug_green", "review_green"}:
        return GRID_LINE_REVIEW_COLOR
    return GRID_LINE_COLOR


def _draw_tick_grid_line(
    canvas,
    start,
    end,
    *,
    color=None,
    thickness=None,
    dash_length=None,
    dash_gap=None,
):
    # 按 2px 实线 + 2px 空白绘制水平/垂直虚线网格。
    # 这里仅负责线条样式，不决定哪些 tick 需要绘制。
    x1, y1 = int(start[0]), int(start[1])
    x2, y2 = int(end[0]), int(end[1])
    color = _active_grid_line_color() if color is None else color
    thickness = GRID_LINE_THICKNESS if thickness is None else int(thickness)
    dash_length = GRID_DASH_LENGTH if dash_length is None else int(dash_length)
    dash_gap = GRID_DASH_GAP if dash_gap is None else int(dash_gap)
    dash_cycle = max(1, dash_length + dash_gap)

    if x1 == x2:
        step = 1 if y2 >= y1 else -1
        for y in range(y1, y2 + step, step * dash_cycle):
            y_end = y + step * (dash_length - 1)
            if step > 0:
                y_end = min(y_end, y2)
            else:
                y_end = max(y_end, y2)
            cv2.line(canvas, (x1, y), (x2, y_end), color, thickness, cv2.LINE_8)
        return

    if y1 == y2:
        step = 1 if x2 >= x1 else -1
        for x in range(x1, x2 + step, step * dash_cycle):
            x_end = x + step * (dash_length - 1)
            if step > 0:
                x_end = min(x_end, x2)
            else:
                x_end = max(x_end, x2)
            cv2.line(canvas, (x, y1), (x_end, y2), color, thickness, cv2.LINE_8)
        return

    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_8)


def draw_basic_grid(
    img,
    x_pixels,
    y_pixels,
    x_axis,
    y_axis,
    include_axes=False,
    *,
    x_grid_bounds=None,
    y_grid_bounds=None,
    grid_line_color=None,
):
    """
    绘制基础网格 - 只延伸短横线形成网格图
    """
    canvas = img.copy()
    if include_axes:
        cv2.line(canvas, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
        cv2.line(canvas, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (255, 0, 0), 2)
    
    # 绘制水平网格线（Y方向）
    x_min, x_max = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    y_min, y_max = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])
    
    # 绘制垂直网格线
    if isinstance(x_grid_bounds, (list, tuple)) and len(x_grid_bounds) >= 2:
        y_min, y_max = int(x_grid_bounds[0]), int(x_grid_bounds[1])
    if isinstance(y_grid_bounds, (list, tuple)) and len(y_grid_bounds) >= 2:
        x_min, x_max = int(y_grid_bounds[0]), int(y_grid_bounds[1])
    line_color = grid_line_color or GRID_LINE_COLOR
    for x_pix in x_pixels:
        x_pos = int(round(float(x_pix)))
        _draw_tick_grid_line(
            canvas,
            (x_pos, y_min),
            (x_pos, y_max),
            color=line_color,
            thickness=1,
            dash_length=2,
            dash_gap=2,
        )
    
    # 绘制水平网格线
    for y_pix in y_pixels:
        y_pos = int(round(float(y_pix)))
        _draw_tick_grid_line(
            canvas,
            (x_min, y_pos),
            (x_max, y_pos),
            color=line_color,
            thickness=1,
            dash_length=2,
            dash_gap=2,
        )
    
    return canvas

def calculate_max_decimal_places(ticks):
    """
    计算一组刻度的最大小数位数
    对输入的所有刻度值，先消除误差，再计算最大小数位
    """
    max_decimal = 0
    for tick in ticks:
        if isinstance(tick, (int, float)) or (isinstance(tick, str) and tick.replace('.', '', 1).isdigit()):
            try:
                # 先转换为浮点数
                if isinstance(tick, str):
                    num = float(tick)
                else:
                    num = tick
                # 消除浮点数误差
                formatted = f"{num:.12f}"
                # 计算小数位数
                decimal_places = count_decimal_places(formatted)
                max_decimal = max(max_decimal, decimal_places)
            except:
                pass
    return max_decimal

def format_tick_value(value, decimal_places):
    """
    格式化刻度值，保留指定的小数位数
    关键：使用字符串格式化消除浮点数误差
    """
    logger.debug(f"格式化前原始值: {value}, 目标小数位数: {decimal_places}")
    
    try:
        # 转换为浮点数
        if isinstance(value, str):
            num = float(value)
        else:
            num = float(value)
        
        # 第一步：误差消除
        # 使用字符串格式化到12位小数，再四舍五入到decimal_places+1位
        num = round(float(f"{num:.12f}"), decimal_places + 1)
        logger.debug(f"误差消除后: {num}")
        
        # 第二步：强制格式化到指定小数位数
        if decimal_places > 0:
            formatted = f"{num:.{decimal_places}f}"
        else:
            formatted = f"{int(round(num))}"
        
        logger.debug(f"格式化后: {formatted}")
        return formatted
    except Exception as e:
        logger.error(f"格式化刻度值时出错: {e}")
        # 文字轴或其他类型，直接返回
        return str(value)

def draw_encrypted_grid(
    img,
    x_pixels,
    y_pixels,
    x_ticks_encrypted,
    y_ticks_encrypted,
    x_axis,
    y_axis,
    x_axis_type=NUMERIC_AXIS_TYPE,
    y_axis_type=NUMERIC_AXIS_TYPE,
    base_x_pixels=None,
    base_y_pixels=None,
    base_grid_img=None,
    x_grid_bounds=None,
    y_grid_bounds=None,
    x_label_styles=None,
    y_label_styles=None,
    grid_line_color=None,
):
    # 加密图 = 基础网格 + 数值轴插入 tick 的网格线和红色标签。
    # 文字轴不插值，因此不会额外绘制“中间类别”。
    """
    绘制加密网格 - 在基础网格上添加加密刻度线、文本框和文本
    只对数值轴加密生成的部分添加网格线、文本框和文本
    文字轴不加密
    """
    if base_grid_img is not None:
        canvas = base_grid_img.copy()
    else:
        canvas = draw_basic_grid(
            img,
            base_x_pixels if base_x_pixels is not None else x_pixels,
            base_y_pixels if base_y_pixels is not None else y_pixels,
            x_axis,
            y_axis,
            x_grid_bounds=x_grid_bounds,
            y_grid_bounds=y_grid_bounds,
            grid_line_color=GRID_LINE_COLOR,
        )

    # Draw encrypted grid lines only at inserted midpoint positions. Original
    # grid lines are already rendered by draw_basic_grid above.
    x_min, x_max = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    y_min, y_max = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])
    if isinstance(x_grid_bounds, (list, tuple)) and len(x_grid_bounds) >= 2:
        y_min, y_max = int(x_grid_bounds[0]), int(x_grid_bounds[1])
    if isinstance(y_grid_bounds, (list, tuple)) and len(y_grid_bounds) >= 2:
        x_min, x_max = int(y_grid_bounds[0]), int(y_grid_bounds[1])
    x_axis_type = normalize_axis_type_name(x_axis_type)
    y_axis_type = normalize_axis_type_name(y_axis_type)

    def inserted_tick_indices(encrypted_pixels, native_pixels):
        if not isinstance(encrypted_pixels, (list, tuple)):
            return []
        if isinstance(native_pixels, (list, tuple)) and native_pixels:
            native_positions = []
            for pixel in native_pixels:
                try:
                    native_positions.append(int(round(float(pixel))))
                except (TypeError, ValueError):
                    continue
            if native_positions:
                result = []
                for index, pixel in enumerate(encrypted_pixels):
                    try:
                        encrypted_position = int(round(float(pixel)))
                    except (TypeError, ValueError):
                        continue
                    if all(abs(encrypted_position - native_position) > 1 for native_position in native_positions):
                        result.append(index)
                return result
        return [index for index in range(len(encrypted_pixels)) if index % 2 == 1]

    x_inserted_indices = inserted_tick_indices(x_pixels, base_x_pixels)
    y_inserted_indices = inserted_tick_indices(y_pixels, base_y_pixels)

    drawn_x_grid_lines = 0
    if x_axis_type == NUMERIC_AXIS_TYPE:
        for i in x_inserted_indices:
            x_pix = x_pixels[i]
            _draw_tick_grid_line(
                canvas,
                (int(x_pix), y_min),
                (int(x_pix), y_max),
                color=grid_line_color or GRID_LINE_COLOR,
                thickness=1,
                dash_length=2,
                dash_gap=2,
            )
            drawn_x_grid_lines += 1

    drawn_y_grid_lines = 0
    if y_axis_type == NUMERIC_AXIS_TYPE:
        for i in y_inserted_indices:
            y_pix = y_pixels[i]
            _draw_tick_grid_line(
                canvas,
                (x_min, int(y_pix)),
                (x_max, int(y_pix)),
                color=grid_line_color or GRID_LINE_COLOR,
                thickness=1,
                dash_length=2,
                dash_gap=2,
            )
            drawn_y_grid_lines += 1

    logger.debug(f"成功绘制加密网格线: X轴{drawn_x_grid_lines}条, Y轴{drawn_y_grid_lines}条")
    
    # 绘制加密刻度文本标签，优化显示效果
    try:
        # 优化文本样式，减小字体大小避免重叠
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3  # 减小字体大小
        thickness = 1  # 细线
        padding = 3  # 减小内边距
        
        logger.debug(f"准备绘制加密刻度文本: X轴像素点数量={len(x_pixels)}, 加密刻度数量={len(x_ticks_encrypted)}")
        logger.debug(f"Y轴像素点数量={len(y_pixels)}, 加密刻度数量={len(y_ticks_encrypted)}")
        logger.debug(f"X轴类型: {x_axis_type}, Y轴类型: {y_axis_type}")
        
        # 计算X轴的最大小数位数
        x_max_decimal = calculate_max_decimal_places(x_ticks_encrypted)
        logger.debug(f"X轴最大小数位数: {x_max_decimal}")
        
        # 计算Y轴的最大小数位数
        y_max_decimal = calculate_max_decimal_places(y_ticks_encrypted)
        logger.debug(f"Y轴最大小数位数: {y_max_decimal}")
        
        # 为X轴绘制加密刻度文本（只对数字轴加密部分）
        drawn_x_texts = 0
        if x_axis_type == NUMERIC_AXIS_TYPE:
            x_min, x_max_val = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
            x_axis_y = max(y_axis[1], y_axis[3])  # X轴的Y坐标
            
            # 确保x_pixels和x_ticks_encrypted长度匹配
            if len(x_pixels) == len(x_ticks_encrypted):
                # 加密刻度是在原始刻度之间插入的，所以偶数索引是原始刻度，奇数索引是加密生成的
                for i in x_inserted_indices:
                    # 只处理加密生成的刻度（奇数索引）
                    if True:
                        x_pix = x_pixels[i]
                        # 检查索引是否有效
                        if i < len(x_ticks_encrypted):
                            tick_value = x_ticks_encrypted[i]
                            
                            # 确保值有效并格式化
                            if tick_value is not None:
                                # 格式化刻度值，保留X轴的最大小数位数
                                text = format_tick_value(tick_value, x_max_decimal)
                                
                                # 获取文本大小
                                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                                
                                # 计算文本位置 - 放在X轴上方，确保足够空间
                                text_x = int(round(x_pix - text_size[0] / 2))
                                text_x = int(np.clip(
                                    text_x,
                                    padding,
                                    max(padding, canvas.shape[1] - text_size[0] - padding),
                                ))
                                # 确保文本在X轴上方有足够空间，避免重叠
                                # Keep encrypted labels visible when the axis is on the canvas edge.
                                if x_axis_y > canvas.shape[0] * 0.65:
                                    y_candidates = [x_axis_y - 8, x_axis_y + text_size[1] + 16]
                                else:
                                    y_candidates = [x_axis_y + text_size[1] + 16, x_axis_y - 8]
                                text_y = None
                                for candidate_y in y_candidates:
                                    candidate_y = int(round(candidate_y))
                                    if (
                                        0 <= candidate_y - text_size[1] - padding
                                        and candidate_y + padding <= canvas.shape[0]
                                    ):
                                        text_y = candidate_y
                                        break
                                if text_y is None:
                                    text_y = int(np.clip(
                                        y_candidates[0],
                                        text_size[1] + padding,
                                        canvas.shape[0] - padding,
                                    ))
                                
                                # 边界检查，确保不与图表内容重叠
                                if (0 <= text_x and text_x + text_size[0] <= canvas.shape[1] and \
                                   0 <= text_y - text_size[1] - padding and text_y + padding <= canvas.shape[0]):
                                    # 使用半透明背景，减少对图表的遮挡
                                    label_color = (0, 0, 0)
                                    # 添加透明度
                                    # 添加细边框
                                    # 绘制红色文本
                                    label_style = None
                                    if isinstance(x_label_styles, list) and i < len(x_label_styles):
                                        label_style = x_label_styles[i]
                                    drawn_like_ocr = _draw_ocr_style_label(
                                        canvas,
                                        img,
                                        text,
                                        label_style,
                                        label_color,
                                    )
                                    if not drawn_like_ocr:
                                        _draw_enhanced_style_label(
                                            canvas,
                                            text,
                                            (int(round(x_pix)), int(round(text_y))),
                                            label_color,
                                            anchor="center",
                                        )
                                    drawn_x_texts += 1
        
        # 为Y轴绘制加密刻度文本（只对数字轴加密部分）
        drawn_y_texts = 0
        if y_axis_type == NUMERIC_AXIS_TYPE:
            y_min, y_max_val = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])
            y_axis_x = _axis_x(y_axis)
            y_label_on_right = y_axis_x > canvas.shape[1] * 0.5
            
            # 确保y_pixels和y_ticks_encrypted长度匹配
            if len(y_pixels) == len(y_ticks_encrypted):
                # 加密刻度是在原始刻度之间插入的，所以偶数索引是原始刻度，奇数索引是加密生成的
                for i in y_inserted_indices:
                    # 只处理加密生成的刻度（奇数索引）
                    if True:
                        y_pix = y_pixels[i]
                        # 检查索引是否有效
                        if i < len(y_ticks_encrypted):
                            tick_value = y_ticks_encrypted[i]
                            
                            # 确保值有效并格式化
                            if tick_value is not None:
                                # 格式化刻度值，保留Y轴的最大小数位数
                                text = format_tick_value(tick_value, y_max_decimal)
                                
                                # 获取文本大小
                                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                                
                                if y_label_on_right:
                                    text_x = y_axis_x + 8
                                    if text_x + text_size[0] + padding > canvas.shape[1]:
                                        text_x = y_axis_x - text_size[0] - 8
                                else:
                                    text_x = y_axis_x - text_size[0] - 8
                                    if text_x - padding < 0:
                                        text_x = y_axis_x + 8
                                text_y = y_pix + text_size[1] // 2
                                
                                # 边界检查，确保不与图表内容重叠
                                chart_content_margin = 50  # 图表内容边缘距离
                                if (0 <= text_y and text_y - text_size[1] - padding >= 0 and \
                                   text_x - padding >= 0 and text_x + text_size[0] + padding <= canvas.shape[1] and \
                                   text_x <= canvas.shape[1] - chart_content_margin):  # 确保在图表内容左侧
                                    # 使用半透明背景，减少对图表的遮挡
                                    label_color = (0, 0, 0)
                                    anchor = "left" if text_x >= x_min else "right"
                                    # 添加透明度
                                    # 添加细边框
                                    # 绘制红色文本
                                    label_style = None
                                    if isinstance(y_label_styles, list) and i < len(y_label_styles):
                                        label_style = y_label_styles[i]
                                    drawn_like_ocr = _draw_ocr_style_label(
                                        canvas,
                                        img,
                                        text,
                                        label_style,
                                        label_color,
                                    )
                                    if not drawn_like_ocr:
                                        _draw_enhanced_style_label(
                                            canvas,
                                            text,
                                            (int(round(text_x)), int(round(text_y))),
                                            label_color,
                                            anchor=anchor,
                                        )
                                    drawn_y_texts += 1
        
        logger.debug(f"成功绘制加密刻度文本: X轴{drawn_x_texts}个, Y轴{drawn_y_texts}个")
        
        # 不再添加水印，避免干扰图表
                    
    except Exception as e:
        logger.error(f"绘制加密刻度文本时出错: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
    
    return canvas

def count_decimal_places(value):
    """
    计算浮点数的小数位数
    改用字符串方式计算，处理浮点数误差
    """
    try:
        # 转换为浮点数
        if isinstance(value, str):
            value = float(value)
        
        # 先消除浮点数误差
        s = f"{value:.12f}"
        
        # 处理科学计数法
        if 'e' in s.lower():
            # 转换为普通小数形式
            parts = s.lower().split('e')
            num = float(parts[0])
            exp = int(parts[1])
            s = f"{num * (10 ** exp):.12f}"
        
        # 去掉末尾的0
        if '.' in s:
            # 分割整数和小数部分
            int_part, dec_part = s.split('.')
            # 去掉小数部分末尾的0
            dec_part = dec_part.rstrip('0')
            # 如果小数部分为空，返回0
            if not dec_part:
                return 0
            # 返回小数部分长度
            return len(dec_part)
        return 0
    except:
        return 0

def process_chart(
    image_path,
    output_dir,
    chart_type_override=None,
    chart_id_override=None,
    axis_repair_hint=None,
    disable_cache=False,
):
    # 处理单张直角系图表，生成基础网格、加密网格和 tick sidecar JSON。
    # axis_repair_hint 来自上传阶段 MLLM，用于指导缺轴、弱轴、右侧轴和背景网格恢复。
    # 正常图表不会因为该参数存在而强制补轴；所有修复逻辑都带保护条件。
    """
    处理单个图表，生成两种网格图像和刻度信息
    1. _grid: 基础网格 - 短横线延伸形成网格图
    2. _with_grid: 加密网格 - 在基础网格上添加加密刻度和文本
    """
    logger.info(f"处理图像: {image_path}")
    logger.debug(f"输出目录: {output_dir}")
    chart_id = chart_id_override or os.path.splitext(os.path.basename(image_path))[0]
    chart_type = (chart_type_override or os.path.basename(os.path.dirname(image_path))).lower()
    axis_repair_hint = normalize_axis_repair_hint(axis_repair_hint)
    repair_applied = {
        "x_axis": False,
        "y_axis": False,
        "x_ticks": False,
        "y_ticks": False,
        "hint": axis_repair_hint,
    }
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 阶段 1：加载图像。使用 np.fromfile + cv2.imdecode 兼容 Windows 中文路径。
    # 加载图像
    try:
        # 修复Windows路径问题 - 使用绝对路径和规范化
        image_path = os.path.abspath(os.path.normpath(image_path))
        logger.debug(f"规范化后的绝对图像路径: {image_path}")
        
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return None
        
        # 检查文件大小
        file_size = os.path.getsize(image_path)
        logger.debug(f"图像文件大小: {file_size} 字节")
        
        # 在Windows上，尝试不同的编码方式
        # 使用双反斜杠确保路径正确
        alt_path = image_path.replace('/', '\\')
        logger.debug(f"Windows格式路径: {alt_path}")
        
        # 尝试使用numpy和cv2.imdecode处理中文路径问题
        try:
            import numpy as np
            # 读取文件内容
            img_data = np.fromfile(image_path, dtype=np.uint8)
            # 解码图像
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            if img is not None:
                logger.info(f"成功加载图像，形状: {img.shape}")
            else:
                logger.error(f"cv2.imdecode失败，但文件存在且可读")
                logger.debug(f"文件头: {img_data[:12].hex()}")
                return None
        except Exception as e:
            logger.error(f"使用numpy和cv2.imdecode加载图像时出错: {str(e)}")
            return None
        
        h, w = img.shape[:2]
        if not disable_cache:
            fast_refreshed = _try_fast_refresh_encrypted_grid_from_existing_outputs(
                img,
                image_path,
                output_dir,
                chart_id,
            )
            if fast_refreshed is not None:
                return fast_refreshed
        enhanced_grid_first_result = _process_chart_with_enhanced_grid_only(
            img,
            image_path,
            output_dir,
            chart_id,
            chart_type,
            axis_repair_hint=axis_repair_hint,
            disable_cache=disable_cache,
        )
        if enhanced_grid_first_result is not None:
            return enhanced_grid_first_result
        logger.error(
            "Enhanced grid-first cartesian flow failed for %s; legacy CV cartesian flow is disabled.",
            image_path,
        )
        return None
        logger.debug(f"图像尺寸: {w}x{h}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 阶段 2：检测候选直线。后续坐标轴、背景网格和边框判断都基于这些线段。
        # 1. 检测直线，调整参数以提高检测率
        logger.debug("开始检测直线...")
        raw_lines = detect_candidate_lines(
            gray, 
            canny_threshold1=30,  # 降低阈值以检测更多边缘
            canny_threshold2=100,
            hough_threshold=15,    # 降低阈值
            min_length=15,         # 缩短最小长度
            max_gap=15             # 增加最大间隙
        )
        logger.debug(f"检测到 {len(raw_lines)} 条原始直线")
        
        if not raw_lines:
            logger.warning(f"未检测到直线: {image_path}")
            return None
        
        # 合并相邻且方向一致的线段，减少双线、碎线对轴推断的干扰。
        # 2. 合并相似直线
        merged_lines = merge_similar_lines(raw_lines)
        logger.debug(f"合并后得到 {len(merged_lines)} 条直线")
        
        # 阶段 3：坐标轴初判。先走通用轴推断，后续再按图表类型修正。
        # 3. 推断坐标轴
        logger.debug("开始推断坐标轴...")
        try:
            x_axis, y_axis, _ = infer_axes_from_lines(merged_lines, (w, h), gray)
        except Exception as e:
            logger.error(f"推断坐标轴时出错: {e}")
            # 尝试直接检测坐标轴
            x_axis, y_axis = None, None
            
            # 检测底部的水平线作为X轴
            for line in merged_lines:
                x1, y1, x2, y2 = line
                # 水平线且接近底部
                if abs(y1 - y2) < 10 and max(y1, y2) > h * 0.7:
                    if x_axis is None or (max(x2, x1) - min(x1, x2)) > (x_axis[2] - x_axis[0]):
                        x_axis = line
            
            # 检测左侧的垂直线作为Y轴
            for line in merged_lines:
                x1, y1, x2, y2 = line
                # 垂直线且接近左侧
                if abs(x1 - x2) < 10 and min(x1, x2) < w * 0.3:
                    if y_axis is None or (max(y2, y1) - min(y1, y2)) > (y_axis[3] - y_axis[1]):
                        y_axis = line
        
        if chart_type == "line" and (x_axis is None or y_axis is None):
            x_axis, y_axis, line_axes_repaired = infer_line_axes_from_visual_structure(
                img,
                merged_lines,
                x_axis,
                y_axis,
            )
            if line_axes_repaired:
                repair_applied["line_axes_inferred_from_visual_structure"] = True
                logger.debug(
                    "Line axes inferred from visual structure: X轴=%s, Y轴=%s",
                    x_axis,
                    y_axis,
                )

        # 阶段 4：缺轴/弱轴修复。把 CV 失败信息合并到上传先验，再按图表类型修复轴线。
        before_repair_x_axis, before_repair_y_axis = x_axis, y_axis
        repair_boxes = _bar_boxes(img, chart_type) if _is_bar_chart_type(chart_type) else []
        axis_repair_hint = _mark_missing_axis_from_cv(
            chart_type,
            axis_repair_hint,
            x_axis,
            y_axis,
            repair_boxes,
        )
        repair_applied["hint"] = axis_repair_hint
        if chart_type in {"scatter", "bubble"} and (
            x_axis is None
            or y_axis is None
            or axis_repair_hint.get("x_axis_missing")
            or axis_repair_hint.get("y_axis_missing")
        ):
            x_axis, y_axis, point_axes_inferred = infer_point_axes_from_visual_structure(
                img,
                merged_lines,
                chart_type,
                x_axis,
                y_axis,
                axis_repair_hint,
            )
            if point_axes_inferred:
                repair_applied["point_axes_inferred_from_visual_structure"] = True
                logger.debug(
                    "Point-chart axes inferred from visual structure: X轴=%s, Y轴=%s",
                    x_axis,
                    y_axis,
                )
        if axis_repair_enabled(axis_repair_hint):
            x_axis, y_axis, repair_boxes = repair_missing_axes(
                img,
                merged_lines,
                x_axis,
                y_axis,
                chart_type,
                axis_repair_hint,
            )
            repair_applied["x_axis"] = axis_repair_hint.get("x_axis_missing") and x_axis is not None
            repair_applied["y_axis"] = axis_repair_hint.get("y_axis_missing") and y_axis is not None
        if x_axis is None or y_axis is None:
            logger.warning(f"未检测到 X/Y 轴: {image_path}")
            return None
        x_axis = _clamp_axis_line(x_axis, img.shape)
        y_axis = _clamp_axis_line(y_axis, img.shape)
        
        logger.debug(f"检测到坐标轴: X轴={x_axis}, Y轴={y_axis}")
        
        # 4. 检测刻度线
        logger.debug("开始检测刻度线...")
        x_axis, y_axis, point_axes_refined = refine_point_chart_axes_from_gridlines(
            merged_lines, x_axis, y_axis, chart_type, img.shape
        )
        if point_axes_refined:
            x_axis = _clamp_axis_line(x_axis, img.shape)
            y_axis = _clamp_axis_line(y_axis, img.shape)
            repair_applied["axis_refined_from_gridlines"] = True
            logger.debug(
                "Point-chart axes refined from gridlines: X轴=%s, Y轴=%s",
                x_axis,
                y_axis,
            )
        elif chart_type in {"scatter", "bubble"} and _point_chart_grid_hint(axis_repair_hint):
            bootstrap_x_axis, bootstrap_y_axis, bootstrap_axes_refined = refine_point_chart_axes_for_bootstrap(
                img,
                x_axis,
                y_axis,
            )
            if bootstrap_axes_refined:
                x_axis, y_axis = bootstrap_x_axis, bootstrap_y_axis
                x_axis = _clamp_axis_line(x_axis, img.shape)
                y_axis = _clamp_axis_line(y_axis, img.shape)
                repair_applied["axis_refined_from_bootstrap_grid"] = True
                logger.debug(
                    "Point-chart axes refined from bootstrap grid projection: X轴=%s, Y轴=%s",
                    x_axis,
                    y_axis,
                )

        x_axis, y_axis, explicit_point_axes_extended = refine_explicit_point_axes_from_strokes(
            img,
            x_axis,
            y_axis,
            chart_type,
            axis_repair_hint,
        )
        if explicit_point_axes_extended:
            x_axis = _clamp_axis_line(x_axis, img.shape)
            y_axis = _clamp_axis_line(y_axis, img.shape)
            repair_applied["explicit_point_axes_extended_from_strokes"] = True
            logger.debug(
                "Explicit point-chart axes extended from strokes: X轴=%s, Y轴=%s",
                x_axis,
                y_axis,
            )

        # 阶段 5：检测短 tick。若短 tick 不足，line/point/bar 会分别从网格线、轴线或柱体中心 bootstrap。
        x_raw_ticks = scan_pixels_for_ticks(img, x_axis, direction='x', scan_range=20)
        y_raw_ticks = scan_pixels_for_ticks(img, y_axis, direction='y', scan_range=20)
        if chart_type == "line":
            if len(x_raw_ticks) < 2:
                x_grid_pixels = infer_tick_pixels_from_gridlines(
                    merged_lines, x_axis, y_axis, "x", img.shape
                )
                if len(x_grid_pixels) < 2:
                    x_grid_pixels = infer_point_chart_grid_pixels_by_projection(
                        img, x_axis, y_axis, "x"
                    )
                if len(x_grid_pixels) < 2:
                    x_grid_pixels = _linspace_pixels_on_axis("x", x_axis, 5)
                if len(x_grid_pixels) >= 2:
                    x_raw_ticks = ticks_from_pixels(x_grid_pixels, x_axis, "x")
                    repair_applied["x_ticks_bootstrapped_for_line"] = True
            if len(y_raw_ticks) < 2:
                y_grid_pixels = infer_tick_pixels_from_gridlines(
                    merged_lines, x_axis, y_axis, "y", img.shape
                )
                if len(y_grid_pixels) < 2:
                    y_grid_pixels = infer_point_chart_grid_pixels_by_projection(
                        img, x_axis, y_axis, "y"
                    )
                if len(y_grid_pixels) < 2:
                    y_grid_pixels = _linspace_pixels_on_axis("y", y_axis, 5)
                if len(y_grid_pixels) >= 2:
                    y_raw_ticks = ticks_from_pixels(y_grid_pixels, y_axis, "y")
                    repair_applied["y_ticks_bootstrapped_for_line"] = True
        if chart_type in {"scatter", "bubble"}:
            if len(x_raw_ticks) < 2:
                if _point_chart_grid_hint(axis_repair_hint) or axis_repair_enabled(axis_repair_hint):
                    x_grid_pixels = bootstrap_point_chart_tick_pixels(
                        img, merged_lines, x_axis, y_axis, "x"
                    )
                else:
                    x_grid_pixels = _linspace_pixels_on_axis("x", x_axis, 5)
                if len(x_grid_pixels) >= 2:
                    x_raw_ticks = ticks_from_pixels(x_grid_pixels, x_axis, "x")
                    repair_applied["x_ticks_bootstrapped"] = True
                    logger.debug(
                        "Bootstrapped %s X tick positions for %s chart.",
                        len(x_grid_pixels),
                        chart_type,
                    )
            if len(y_raw_ticks) < 2:
                if _point_chart_grid_hint(axis_repair_hint) or axis_repair_enabled(axis_repair_hint):
                    y_grid_pixels = bootstrap_point_chart_tick_pixels(
                        img, merged_lines, x_axis, y_axis, "y"
                    )
                else:
                    y_grid_pixels = _linspace_pixels_on_axis("y", y_axis, 5)
                if len(y_grid_pixels) >= 2:
                    y_raw_ticks = ticks_from_pixels(y_grid_pixels, y_axis, "y")
                    repair_applied["y_ticks_bootstrapped"] = True
                    logger.debug(
                        "Bootstrapped %s Y tick positions for %s chart.",
                        len(y_grid_pixels),
                        chart_type,
                    )
        if _is_bar_chart_type(chart_type) and not repair_boxes:
            repair_boxes = _bar_boxes(img, chart_type)
        geometry_hint = apply_bar_geometry_repair_hint(
            chart_type,
            axis_repair_hint,
            repair_boxes,
            x_tick_count=_tick_group_count(x_raw_ticks, "x"),
            y_tick_count=_tick_group_count(y_raw_ticks, "y"),
        )
        if axis_repair_enabled(geometry_hint) and geometry_hint != axis_repair_hint:
            axis_repair_hint = geometry_hint
            repair_applied["hint"] = axis_repair_hint
            x_axis, y_axis, repair_boxes = repair_missing_axes(
                img,
                merged_lines,
                x_axis,
                y_axis,
                chart_type,
                axis_repair_hint,
            )
            repair_applied["x_axis"] = axis_repair_hint.get("x_axis_missing") and x_axis is not None
            repair_applied["y_axis"] = axis_repair_hint.get("y_axis_missing") and y_axis is not None
            logger.debug(
                "Bar geometry repair enabled: x_axis_missing=%s y_axis_missing=%s x_ticks_missing=%s y_ticks_missing=%s",
                axis_repair_hint.get("x_axis_missing"),
                axis_repair_hint.get("y_axis_missing"),
                axis_repair_hint.get("x_ticks_missing"),
                axis_repair_hint.get("y_ticks_missing"),
            )
        logger.debug(f"检测到 X轴刻度 {len(x_raw_ticks)} 个, Y轴刻度 {len(y_raw_ticks)} 个")
        
        if axis_repair_enabled(axis_repair_hint):
            if axis_repair_hint.get("x_ticks_missing"):
                x_pixels = synthesize_tick_pixels_for_missing_axis(
                    chart_type, "x", x_axis, repair_boxes, [], axis_repair_hint
                )
                if len(x_pixels) >= _required_tick_count(chart_type, "x"):
                    x_raw_ticks = ticks_from_pixels(x_pixels, x_axis, "x")
                    repair_applied["x_ticks"] = bool(x_raw_ticks)
            if axis_repair_hint.get("y_ticks_missing"):
                y_pixels = synthesize_tick_pixels_for_missing_axis(
                    chart_type, "y", y_axis, repair_boxes, [], axis_repair_hint
                )
                if len(y_pixels) >= _required_tick_count(chart_type, "y"):
                    y_raw_ticks = ticks_from_pixels(y_pixels, y_axis, "y")
                    repair_applied["y_ticks"] = bool(y_raw_ticks)

        if (
            (len(x_raw_ticks) < _required_tick_count(chart_type, "x") and not axis_repair_hint.get("x_ticks_missing"))
            or (len(y_raw_ticks) < _required_tick_count(chart_type, "y") and not axis_repair_hint.get("y_ticks_missing"))
        ):
            logger.warning(f"未检测到足够的刻度线: {image_path}")
            return None
        
        # 阶段 6：合并和过滤 tick 候选，减少双线、碎片和过密候选。
        # 5. 合并和过滤刻度线
        x_merged_ticks = merge_similar_lines(x_raw_ticks, angle_threshold=np.deg2rad(10))
        y_merged_ticks = merge_similar_lines(y_raw_ticks, angle_threshold=np.deg2rad(10))
        
        x_filtered_ticks = filter_ticks(x_merged_ticks, direction='x')
        y_filtered_ticks = filter_ticks(y_merged_ticks, direction='y')

        if chart_type in {"scatter", "bubble"}:
            if len(x_filtered_ticks) < _required_tick_count(chart_type, "x"):
                if _point_chart_grid_hint(axis_repair_hint) or axis_repair_enabled(axis_repair_hint):
                    x_grid_pixels = bootstrap_point_chart_tick_pixels(
                        img, merged_lines, x_axis, y_axis, "x"
                    )
                else:
                    x_grid_pixels = _linspace_pixels_on_axis("x", x_axis, 5)
                if len(x_grid_pixels) >= _required_tick_count(chart_type, "x"):
                    x_filtered_ticks = ticks_from_pixels(x_grid_pixels, x_axis, "x")
                    repair_applied["x_ticks_bootstrapped_after_filter"] = True
            if len(y_filtered_ticks) < _required_tick_count(chart_type, "y"):
                if _point_chart_grid_hint(axis_repair_hint) or axis_repair_enabled(axis_repair_hint):
                    y_grid_pixels = bootstrap_point_chart_tick_pixels(
                        img, merged_lines, x_axis, y_axis, "y"
                    )
                else:
                    y_grid_pixels = _linspace_pixels_on_axis("y", y_axis, 5)
                if len(y_grid_pixels) >= _required_tick_count(chart_type, "y"):
                    y_filtered_ticks = ticks_from_pixels(y_grid_pixels, y_axis, "y")
                    repair_applied["y_ticks_bootstrapped_after_filter"] = True
        elif chart_type == "line":
            if len(x_filtered_ticks) < _required_tick_count(chart_type, "x"):
                x_grid_pixels = infer_tick_pixels_from_gridlines(
                    merged_lines, x_axis, y_axis, "x", img.shape
                )
                if len(x_grid_pixels) < _required_tick_count(chart_type, "x"):
                    x_grid_pixels = infer_point_chart_grid_pixels_by_projection(
                        img, x_axis, y_axis, "x"
                    )
                if len(x_grid_pixels) < _required_tick_count(chart_type, "x"):
                    x_grid_pixels = _linspace_pixels_on_axis("x", x_axis, 5)
                if len(x_grid_pixels) >= _required_tick_count(chart_type, "x"):
                    x_filtered_ticks = ticks_from_pixels(x_grid_pixels, x_axis, "x")
                    repair_applied["x_ticks_bootstrapped_for_line_after_filter"] = True
            if len(y_filtered_ticks) < _required_tick_count(chart_type, "y"):
                y_grid_pixels = infer_tick_pixels_from_gridlines(
                    merged_lines, x_axis, y_axis, "y", img.shape
                )
                if len(y_grid_pixels) < _required_tick_count(chart_type, "y"):
                    y_grid_pixels = infer_point_chart_grid_pixels_by_projection(
                        img, x_axis, y_axis, "y"
                    )
                if len(y_grid_pixels) < _required_tick_count(chart_type, "y"):
                    y_grid_pixels = _linspace_pixels_on_axis("y", y_axis, 5)
                if len(y_grid_pixels) >= _required_tick_count(chart_type, "y"):
                    y_filtered_ticks = ticks_from_pixels(y_grid_pixels, y_axis, "y")
                    repair_applied["y_ticks_bootstrapped_for_line_after_filter"] = True

        if (
            _is_bar_category_axis(chart_type, "x")
            and _tick_group_count(x_filtered_ticks, "x") < _tick_group_count(x_merged_ticks, "x")
        ):
            x_filtered_ticks = x_merged_ticks
        if (
            _is_bar_category_axis(chart_type, "y")
            and _tick_group_count(y_filtered_ticks, "y") < _tick_group_count(y_merged_ticks, "y")
        ):
            y_filtered_ticks = y_merged_ticks
        
        logger.debug(f"过滤后: X轴刻度 {len(x_filtered_ticks)} 个, Y轴刻度 {len(y_filtered_ticks)} 个")
        
        if (
            (len(x_filtered_ticks) < _required_tick_count(chart_type, "x") and not axis_repair_hint.get("x_ticks_missing"))
            or (len(y_filtered_ticks) < _required_tick_count(chart_type, "y") and not axis_repair_hint.get("y_ticks_missing"))
        ):
            logger.warning(f"未检测到有效的刻度线: {image_path}")
            return None
    
    except Exception as e:
        logger.error(f"前序处理出错: {e}")
        return None
    
    # 阶段 7：调用 MLLM 读取 tick 标签和轴类型，并读取图例颜色/点对象信息。
    # 6. 使用model_processor.py中的函数提取刻度标签和颜色
    logger.debug("开始使用模型提取刻度标签和颜色...")
    
    # 处理刻度标签
    ticks_result = extract_tick_labels_with_llm(
        image_path,
        cache_dir=None if disable_cache else TICK_LABEL_CACHE_DIR,
        chart_type_override=chart_type,
    )
    if ticks_result.get("api_failed"):
        logger.warning(
            "LLM tick labels unavailable after retries; skip chart instead of using positional fallback: %s",
            ticks_result.get("failure_reason", "unknown"),
        )
        return None

    x_ticks_values = ticks_result.get("x_ticks", [])
    y_ticks_values = ticks_result.get("y_ticks", [])
    x_axis_type = ticks_result.get("x_axis_type", NUMERIC_AXIS_TYPE)
    y_axis_type = ticks_result.get("y_axis_type", NUMERIC_AXIS_TYPE)
    x_ticks_values, x_axis_type = coerce_chart_axis_numeric_ticks(
        chart_type, "x", x_ticks_values, x_axis_type
    )
    y_ticks_values, y_axis_type = coerce_chart_axis_numeric_ticks(
        chart_type, "y", y_ticks_values, y_axis_type
    )
    x_axis_type = normalize_axis_type_name(x_axis_type)
    y_axis_type = normalize_axis_type_name(y_axis_type)
    if _is_bar_chart_type(chart_type):
        hint_roles = normalize_axis_repair_hint(axis_repair_hint)
        if _bar_orientation(chart_type) == "v" and hint_roles.get("x_axis_role") in {"category", "date"}:
            x_axis_type = TEXT_AXIS_TYPE
        if _bar_orientation(chart_type) == "v" and hint_roles.get("y_axis_role") == "numeric":
            y_axis_type = NUMERIC_AXIS_TYPE
        if _bar_orientation(chart_type) == "h" and hint_roles.get("x_axis_role") == "numeric":
            x_axis_type = NUMERIC_AXIS_TYPE
        if _bar_orientation(chart_type) == "h" and hint_roles.get("y_axis_role") in {"category", "date"}:
            y_axis_type = TEXT_AXIS_TYPE
        x_ticks_values, x_secondary_removed = _drop_secondary_group_labels_for_bar_axis(
            chart_type,
            "x",
            x_ticks_values,
        )
        y_ticks_values, y_secondary_removed = _drop_secondary_group_labels_for_bar_axis(
            chart_type,
            "y",
            y_ticks_values,
        )
        if x_secondary_removed:
            repair_applied["x_secondary_group_labels_removed"] = True
        if y_secondary_removed:
            repair_applied["y_secondary_group_labels_removed"] = True
        if (
            _bar_orientation(chart_type) == "h"
            and x_axis_type == NUMERIC_AXIS_TYPE
            and axis_repair_hint.get("x_ticks_missing")
            and _numeric_ticks_from_unit_labels(x_ticks_values) is None
        ):
            x_ticks_values = []
            repair_applied["non_numeric_x_tick_labels_dropped_for_missing_value_axis"] = True
        if (
            _bar_orientation(chart_type) == "v"
            and y_axis_type == NUMERIC_AXIS_TYPE
            and axis_repair_hint.get("y_ticks_missing")
            and _numeric_ticks_from_unit_labels(y_ticks_values) is None
        ):
            y_ticks_values = []
            repair_applied["non_numeric_y_tick_labels_dropped_for_missing_value_axis"] = True

    if (
        chart_type in {"v_bar", "v_stacked_bar"}
        and y_axis_type == NUMERIC_AXIS_TYPE
        and _is_bar_chart_type(chart_type)
    ):
        vbar_axes_refined = False
        refined_x_axis, refined_y_axis, axes_refined = refine_vbar_axes_from_grid_bounds(
            merged_lines,
            x_axis,
            y_axis,
            repair_boxes,
            img.shape,
            axis_repair_hint,
        )
        if axes_refined:
            x_axis, y_axis = refined_x_axis, refined_y_axis
            vbar_axes_refined = True
            repair_applied["axis_refined_from_vbar_grid_bounds"] = True

        refined_x_axis, refined_y_axis, axes_refined = refine_vbar_positive_axes_from_plot_bounds(
            merged_lines,
            x_axis,
            y_axis,
            repair_boxes,
            y_ticks_values,
            img.shape,
            axis_repair_hint,
        )
        if axes_refined:
            x_axis, y_axis = refined_x_axis, refined_y_axis
            vbar_axes_refined = True
            repair_applied["axis_refined_from_vbar_positive_plot_bounds"] = True
        if vbar_axes_refined:
            if axis_repair_hint.get("x_ticks_missing") or x_axis_type != NUMERIC_AXIS_TYPE:
                x_pixels = synthesize_tick_pixels_for_missing_axis(
                    chart_type,
                    "x",
                    x_axis,
                    repair_boxes,
                    x_ticks_values,
                    axis_repair_hint,
                )
                if len(x_pixels) >= _required_tick_count(chart_type, "x"):
                    x_filtered_ticks = ticks_from_pixels(x_pixels, x_axis, "x")
                    repair_applied["x_ticks"] = True
            if axis_repair_hint.get("y_ticks_missing") or y_axis_type == NUMERIC_AXIS_TYPE:
                y_pixels = synthesize_tick_pixels_for_missing_axis(
                    chart_type,
                    "y",
                    y_axis,
                    repair_boxes,
                    y_ticks_values,
                    axis_repair_hint,
                )
                if len(y_pixels) >= _required_tick_count(chart_type, "y"):
                    y_filtered_ticks = ticks_from_pixels(y_pixels, y_axis, "y")
                    repair_applied["y_ticks"] = True
    
    # 处理图例颜色
    colors_data = colors_from_axis_repair_hint(axis_repair_hint, chart_type)
    if not colors_data:
        colors_data = extract_chart_color_items(image_path, chart_type)
    
    logger.debug(f"模型处理结果: X轴{len(x_ticks_values)}个刻度, Y轴{len(y_ticks_values)}个刻度, {len(colors_data)}个颜色")
    
    # 初始化刻度数据列表
    x_ticks_data = []  # 数值轴存储为float，文字轴存储为字符串
    x_pixels_data = []
    y_ticks_data = []  # 数值轴存储为float，文字轴存储为字符串
    y_pixels_data = []
    
    # 计算检测到的刻度线的中心位置
    x_pixel_positions = sorted([(t[0] + t[2]) // 2 for t in x_filtered_ticks])
    y_pixel_positions = sorted([(t[1] + t[3]) // 2 for t in y_filtered_ticks], reverse=True)
    x_axis_scale = "linear"
    y_axis_scale = "linear"

    # 阶段 8：tick 文本与像素绑定。上传先验只在当前识别明显不可靠时才覆盖。
    x_ticks_values, x_axis_type, x_upload_prior_applied = _apply_upload_numeric_axis_prior(
        chart_type,
        "x",
        x_ticks_values,
        x_axis_type,
        x_pixel_positions,
        axis_repair_hint,
    )
    y_ticks_values, y_axis_type, y_upload_prior_applied = _apply_upload_numeric_axis_prior(
        chart_type,
        "y",
        y_ticks_values,
        y_axis_type,
        y_pixel_positions,
        axis_repair_hint,
    )
    if x_upload_prior_applied:
        repair_applied["x_tick_labels_from_upload_prior"] = True
    if y_upload_prior_applied:
        repair_applied["y_tick_labels_from_upload_prior"] = True

    if _is_bar_chart_type(chart_type):
        orientation = _bar_orientation(chart_type)
        needs_bar_value_labels = (
            (
                orientation == "v"
                and y_axis_type == NUMERIC_AXIS_TYPE
                and len(y_ticks_values or []) < 2
                and len(y_pixel_positions or []) < 2
            )
            or (
                orientation == "h"
                and x_axis_type == NUMERIC_AXIS_TYPE
                and len(x_ticks_values or []) < 2
                and len(x_pixel_positions or []) < 2
            )
        )
        if needs_bar_value_labels:
            value_result = extract_bar_value_labels_with_llm(
                image_path,
                cache_dir=None if disable_cache else BAR_VALUE_LABEL_CACHE_DIR,
                chart_type_override=chart_type,
            )
            bar_values = value_result.get("values", [])
            if orientation == "v":
                repaired_values, repaired_pixels = _bar_value_axis_ticks_from_data_labels(
                    chart_type,
                    "y",
                    x_axis,
                    repair_boxes or _bar_boxes(img, chart_type),
                    bar_values,
                )
                if len(repaired_values) >= 2:
                    y_ticks_values = repaired_values
                    y_pixel_positions = repaired_pixels
                    if y_pixel_positions:
                        axis_x = _axis_x(y_axis)
                        y_axis = [axis_x, max(y_pixel_positions), axis_x, min(y_pixel_positions)]
                    y_axis_type = NUMERIC_AXIS_TYPE
                    repair_applied["y_ticks_from_bar_value_labels"] = True
            else:
                repaired_values, repaired_pixels = _bar_value_axis_ticks_from_data_labels(
                    chart_type,
                    "x",
                    y_axis,
                    repair_boxes or _bar_boxes(img, chart_type),
                    bar_values,
                )
                if len(repaired_values) >= 2:
                    x_ticks_values = repaired_values
                    x_pixel_positions = repaired_pixels
                    if x_pixel_positions:
                        axis_y = _axis_y(x_axis)
                        x_axis = [min(x_pixel_positions), axis_y, max(x_pixel_positions), axis_y]
                    x_axis_type = NUMERIC_AXIS_TYPE
                    repair_applied["x_ticks_from_bar_value_labels"] = True

        if (
            orientation == "h"
            and x_axis_type == NUMERIC_AXIS_TYPE
            and axis_repair_hint.get("x_ticks_missing")
            and len(x_ticks_values or []) < 2
            and len(x_pixel_positions or []) < 2
            and x_axis is not None
        ):
            x_ticks_values = [0, 20, 40, 60, 80, 100]
            x_pixel_positions = _linspace_pixels_on_axis("x", x_axis, len(x_ticks_values))
            repair_applied["x_ticks_synthesized_for_no_axis_bar"] = True
        if (
            orientation == "v"
            and y_axis_type == NUMERIC_AXIS_TYPE
            and axis_repair_hint.get("y_ticks_missing")
            and len(y_ticks_values or []) < 2
            and len(y_pixel_positions or []) < 2
            and y_axis is not None
        ):
            y_ticks_values = [0, 20, 40, 60, 80, 100]
            y_pixel_positions = _linspace_pixels_on_axis("y", y_axis, len(y_ticks_values))
            repair_applied["y_ticks_synthesized_for_no_axis_bar"] = True

    if (
        chart_type in {"scatter", "bubble"}
        and axis_repair_enabled(axis_repair_hint)
        and (
            axis_repair_hint.get("x_axis_missing")
            or axis_repair_hint.get("y_axis_missing")
            or axis_repair_hint.get("x_ticks_missing")
            or axis_repair_hint.get("y_ticks_missing")
        )
    ):
        projected_x_pixels = infer_point_chart_grid_pixels_for_missing_axes(img, "x")
        projected_y_pixels = infer_point_chart_grid_pixels_for_missing_axes(img, "y")
        selected_x_pixels, selected_x_scale = select_projected_tick_pixels_for_values(
            projected_x_pixels,
            x_ticks_values,
            "x",
            x_axis,
        )
        selected_y_pixels, selected_y_scale = select_projected_tick_pixels_for_values(
            projected_y_pixels,
            y_ticks_values,
            "y",
            y_axis,
        )
        if len(selected_x_pixels) == len(x_ticks_values or []) and len(selected_x_pixels) >= 2:
            x_pixel_positions = selected_x_pixels
            x_axis_scale = selected_x_scale
            repair_applied["x_ticks"] = True
            repair_applied["x_ticks_refined_from_missing_axis_grid"] = True
            logger.debug(
                "Point-chart missing-axis X ticks selected from projection: %s scale=%s",
                x_pixel_positions,
                x_axis_scale,
            )
        if len(selected_y_pixels) == len(y_ticks_values or []) and len(selected_y_pixels) >= 2:
            y_pixel_positions = selected_y_pixels
            y_axis_scale = selected_y_scale
            repair_applied["y_ticks"] = True
            repair_applied["y_ticks_refined_from_missing_axis_grid"] = True
            logger.debug(
                "Point-chart missing-axis Y ticks selected from projection: %s scale=%s",
                y_pixel_positions,
                y_axis_scale,
            )
        if (
            repair_applied.get("x_ticks_refined_from_missing_axis_grid")
            and repair_applied.get("y_ticks_refined_from_missing_axis_grid")
            and len(x_pixel_positions) >= 2
            and len(y_pixel_positions) >= 2
        ):
            new_x_axis, new_y_axis, axes_refined = refine_point_chart_axes_from_projected_ticks(
                img,
                x_axis,
                y_axis,
                x_pixel_positions,
                y_pixel_positions,
                y_axis_scale,
                prefer_grid_bounds=True,
            )
            if axes_refined:
                x_axis, y_axis = new_x_axis, new_y_axis
                repair_applied["x_axis"] = True
                repair_applied["y_axis"] = True
                repair_applied["axis_refined_from_missing_point_grid"] = True
                repair_applied["plot_bounds_refined_from_horizontal_grid"] = True
                if y_axis_scale != "linear":
                    repair_applied["plot_vertical_bounds_refined_from_grid"] = True

    if _is_bar_chart_type(chart_type) and axis_repair_enabled(axis_repair_hint):
        if x_axis_type == NUMERIC_AXIS_TYPE:
            x_pixel_positions = add_missing_numeric_axis_endpoints(
                "x", x_axis, x_pixel_positions, x_ticks_values
            )
        if y_axis_type == NUMERIC_AXIS_TYPE:
            y_pixel_positions = add_missing_numeric_axis_endpoints(
                "y", y_axis, y_pixel_positions, y_ticks_values
            )

        orientation = _bar_orientation(chart_type)
        if (
            orientation == "v"
            and y_axis_type == NUMERIC_AXIS_TYPE
            and (
                _axis_needs_grid_recovery(axis_repair_hint, "y")
                or axis_repair_hint.get("y_ticks_missing")
                or len(y_pixel_positions) < len(y_ticks_values or [])
                or (
                    len(y_ticks_values or []) >= 2
                    and len(y_pixel_positions) > max(len(y_ticks_values) * 2, len(y_ticks_values) + 3)
                )
            )
        ):
            recovered_pixels, recovered_scale = recover_bar_value_axis_pixels_from_grid(
                img,
                merged_lines,
                x_axis,
                y_axis,
                chart_type,
                "y",
                y_ticks_values,
            )
            if recovered_pixels:
                y_pixel_positions = recovered_pixels
                y_axis_scale = recovered_scale
                repair_applied["y_ticks"] = True
                repair_applied["y_ticks_refined_from_grid"] = True
            elif (
                len(y_ticks_values or []) >= 2
                and len(y_pixel_positions) > max(len(y_ticks_values) * 2, len(y_ticks_values) + 3)
            ):
                y_low, y_high = sorted([int(y_axis[1]), int(y_axis[3])])
                y_pixel_positions = [
                    int(round(value))
                    for value in np.linspace(y_high, y_low, len(y_ticks_values))
                ]
                repair_applied["y_ticks"] = True
                repair_applied["y_ticks_bound_to_label_count"] = True
        elif (
            orientation == "h"
            and x_axis_type == NUMERIC_AXIS_TYPE
            and (
                _axis_needs_grid_recovery(axis_repair_hint, "x")
                or axis_repair_hint.get("x_ticks_missing")
                or len(x_pixel_positions) < len(x_ticks_values or [])
                or (
                    len(x_ticks_values or []) >= 2
                    and len(x_pixel_positions) > max(len(x_ticks_values) * 2, len(x_ticks_values) + 3)
                )
            )
        ):
            recovered_pixels, recovered_scale = recover_bar_value_axis_pixels_from_grid(
                img,
                merged_lines,
                x_axis,
                y_axis,
                chart_type,
                "x",
                x_ticks_values,
            )
            if recovered_pixels:
                x_pixel_positions = recovered_pixels
                x_axis_scale = recovered_scale
                repair_applied["x_ticks"] = True
                repair_applied["x_ticks_refined_from_grid"] = True
            elif (
                len(x_ticks_values or []) >= 2
                and len(x_pixel_positions) > max(len(x_ticks_values) * 2, len(x_ticks_values) + 3)
            ):
                x_start, x_end = sorted([int(x_axis[0]), int(x_axis[2])])
                x_pixel_positions = [
                    int(round(value))
                    for value in np.linspace(x_start, x_end, len(x_ticks_values))
                ]
                repair_applied["x_ticks"] = True
                repair_applied["x_ticks_bound_to_label_count"] = True

    if axis_repair_enabled(axis_repair_hint) and not repair_applied.get("axis_refined_from_missing_point_grid"):
        if axis_repair_hint.get("x_ticks_missing"):
            repaired_x_pixels = []
            if chart_type in {"scatter", "bubble"} and normalize_axis_type_name(x_axis_type) == NUMERIC_AXIS_TYPE:
                point_x_candidates = infer_point_tick_pixels_from_grid_candidates(
                    img,
                    merged_lines,
                    "x",
                )
                selected_x_pixels, selected_x_scale = select_projected_tick_pixels_for_values(
                    point_x_candidates,
                    x_ticks_values,
                    "x",
                    x_axis,
                )
                if len(selected_x_pixels) == len(x_ticks_values or []) and len(selected_x_pixels) >= 2:
                    repaired_x_pixels = selected_x_pixels
                    x_axis_scale = selected_x_scale
                    repair_applied["x_ticks_refined_from_point_grid_candidates"] = True
            if not repaired_x_pixels and normalize_axis_type_name(x_axis_type) == NUMERIC_AXIS_TYPE:
                endpoint_x_pixels = add_missing_numeric_axis_endpoints(
                    "x",
                    x_axis,
                    x_pixel_positions,
                    x_ticks_values,
                )
                if len(endpoint_x_pixels) == len(x_ticks_values or []):
                    repaired_x_pixels = endpoint_x_pixels
            if not repaired_x_pixels:
                repaired_x_pixels = synthesize_tick_pixels_for_missing_axis(
                    chart_type, "x", x_axis, repair_boxes, x_ticks_values, axis_repair_hint
                )
            if len(repaired_x_pixels) >= _required_tick_count(chart_type, "x"):
                x_pixel_positions = repaired_x_pixels
                repair_applied["x_ticks"] = True
        if axis_repair_hint.get("y_ticks_missing"):
            repaired_y_pixels = []
            if chart_type in {"scatter", "bubble"} and normalize_axis_type_name(y_axis_type) == NUMERIC_AXIS_TYPE:
                point_y_candidates = infer_point_tick_pixels_from_grid_candidates(
                    img,
                    merged_lines,
                    "y",
                )
                selected_y_pixels, selected_y_scale = select_projected_tick_pixels_for_values(
                    point_y_candidates,
                    y_ticks_values,
                    "y",
                    y_axis,
                )
                if len(selected_y_pixels) == len(y_ticks_values or []) and len(selected_y_pixels) >= 2:
                    repaired_y_pixels = selected_y_pixels
                    y_axis_scale = selected_y_scale
                    repair_applied["y_ticks_refined_from_point_grid_candidates"] = True
            if not repaired_y_pixels and normalize_axis_type_name(y_axis_type) == NUMERIC_AXIS_TYPE:
                endpoint_y_pixels = add_missing_numeric_axis_endpoints(
                    "y",
                    y_axis,
                    y_pixel_positions,
                    y_ticks_values,
                )
                if len(endpoint_y_pixels) == len(y_ticks_values or []):
                    repaired_y_pixels = endpoint_y_pixels
            if not repaired_y_pixels:
                repaired_y_pixels = synthesize_tick_pixels_for_missing_axis(
                    chart_type, "y", y_axis, repair_boxes, y_ticks_values, axis_repair_hint
                )
            if len(repaired_y_pixels) >= _required_tick_count(chart_type, "y"):
                y_pixel_positions = repaired_y_pixels
                repair_applied["y_ticks"] = True

    if (
        chart_type in {"scatter", "bubble"}
        and repair_applied.get("x_ticks_refined_from_point_grid_candidates")
        and repair_applied.get("y_ticks_refined_from_point_grid_candidates")
        and len(x_pixel_positions) >= 2
        and len(y_pixel_positions) >= 2
    ):
        plot_left = int(min(x_pixel_positions))
        plot_right = int(max(x_pixel_positions))
        plot_top = int(min(y_pixel_positions))
        plot_bottom = int(max(y_pixel_positions))
        if plot_right > plot_left and plot_bottom > plot_top:
            x_axis = [plot_left, plot_bottom, plot_right, plot_bottom]
            y_axis = [plot_left, plot_bottom, plot_left, plot_top]
            repair_applied["axis_refined_from_point_grid_candidates"] = True

    if (
        chart_type in {"scatter", "bubble"}
        and repair_applied.get("axis_refined_from_gridlines")
        and not repair_applied.get("axis_refined_from_missing_point_grid")
        and _point_chart_grid_hint(axis_repair_hint)
    ):
        projected_x_pixels = infer_point_chart_grid_pixels_for_missing_axes(img, "x")
        projected_y_pixels = infer_point_chart_grid_pixels_for_missing_axes(img, "y")
        selected_x_pixels, selected_x_scale = select_projected_tick_pixels_for_values(
            projected_x_pixels,
            x_ticks_values,
            "x",
            x_axis,
        )
        selected_y_pixels, selected_y_scale = select_projected_tick_pixels_for_values(
            projected_y_pixels,
            y_ticks_values,
            "y",
            y_axis,
        )
        if (
            x_axis_type == NUMERIC_AXIS_TYPE
            and y_axis_type == NUMERIC_AXIS_TYPE
            and len(selected_x_pixels) == len(x_ticks_values or [])
            and len(selected_y_pixels) == len(y_ticks_values or [])
            and len(selected_x_pixels) >= 2
            and len(selected_y_pixels) >= 2
        ):
            x_pixel_positions = selected_x_pixels
            y_pixel_positions = selected_y_pixels
            x_axis_scale = selected_x_scale
            y_axis_scale = selected_y_scale
            new_x_axis, new_y_axis, axes_refined = refine_point_chart_axes_from_projected_ticks(
                img,
                x_axis,
                y_axis,
                x_pixel_positions,
                y_pixel_positions,
                y_axis_scale,
                prefer_grid_bounds=bool(
                    repair_applied.get("x_ticks_bootstrapped")
                    or repair_applied.get("y_ticks_bootstrapped")
                    or repair_applied.get("x_ticks_bootstrapped_after_filter")
                    or repair_applied.get("y_ticks_bootstrapped_after_filter")
                    or axis_repair_hint.get("plot_area_style") == "grid_only"
                ),
            )
            if axes_refined:
                x_axis, y_axis = new_x_axis, new_y_axis
                repair_applied["axis_refined_from_projected_tick_grid"] = True
                repair_applied["plot_bounds_refined_from_horizontal_grid"] = True
            repair_applied["x_ticks_refined_from_projection_grid"] = True
            repair_applied["y_ticks_refined_from_projection_grid"] = True
            logger.debug(
                "Point-chart gridline-refined axes bound to projected tick grids: x=%s y=%s x_scale=%s y_scale=%s",
                x_pixel_positions,
                y_pixel_positions,
                x_axis_scale,
                y_axis_scale,
            )
        elif x_axis_type == NUMERIC_AXIS_TYPE and len(x_ticks_values or []) >= 2:
            x_start, x_end = sorted([int(x_axis[0]), int(x_axis[2])])
            x_pixel_positions = [int(round(value)) for value in np.linspace(x_start, x_end, len(x_ticks_values))]
            repair_applied["x_ticks_refined_from_axis"] = True
        if (
            not repair_applied.get("y_ticks_refined_from_projection_grid")
            and y_axis_type == NUMERIC_AXIS_TYPE
            and len(y_ticks_values or []) >= 2
        ):
            y_low, y_high = sorted([int(y_axis[1]), int(y_axis[3])])
            y_pixel_positions = [int(round(value)) for value in np.linspace(y_high, y_low, len(y_ticks_values))]
            repair_applied["y_ticks_refined_from_axis"] = True

    if chart_type in {"scatter", "bubble"}:
        if (
            normalize_axis_type_name(x_axis_type) == NUMERIC_AXIS_TYPE
            and _point_chart_grid_hint(axis_repair_hint)
            and point_chart_tick_pixels_are_suspicious(
                x_pixel_positions, x_axis, "x", len(x_ticks_values or [])
            )
        ):
            projected_x_pixels = infer_point_chart_grid_pixels_by_projection(
                img, x_axis, y_axis, "x", expected_count=len(x_ticks_values or [])
            )
            if len(projected_x_pixels) >= max(2, min(len(x_ticks_values or []), 3)):
                x_pixel_positions = projected_x_pixels
                repair_applied["x_ticks_refined_from_projection_grid"] = True
                logger.debug(
                    "Point-chart X ticks refined from projection grid: %s",
                    x_pixel_positions,
                )
        if (
            normalize_axis_type_name(y_axis_type) == NUMERIC_AXIS_TYPE
            and _point_chart_grid_hint(axis_repair_hint)
            and point_chart_tick_pixels_are_suspicious(
                y_pixel_positions, y_axis, "y", len(y_ticks_values or [])
            )
        ):
            projected_y_pixels = infer_point_chart_grid_pixels_by_projection(
                img, x_axis, y_axis, "y", expected_count=len(y_ticks_values or [])
            )
            if len(projected_y_pixels) >= max(2, min(len(y_ticks_values or []), 3)):
                y_pixel_positions = projected_y_pixels
                repair_applied["y_ticks_refined_from_projection_grid"] = True
                logger.debug(
                    "Point-chart Y ticks refined from projection grid: %s",
                    y_pixel_positions,
                )
        if not _point_chart_grid_hint(axis_repair_hint) and not axis_repair_enabled(axis_repair_hint):
            if normalize_axis_type_name(x_axis_type) == NUMERIC_AXIS_TYPE and len(x_ticks_values or []) >= 2:
                x_pixel_positions = _linspace_pixels_on_axis("x", x_axis, len(x_ticks_values))
                repair_applied["x_ticks_bound_to_axis_for_explicit_point_chart"] = True
            if normalize_axis_type_name(y_axis_type) == NUMERIC_AXIS_TYPE and len(y_ticks_values or []) >= 2:
                y_pixel_positions = _linspace_pixels_on_axis("y", y_axis, len(y_ticks_values))
                repair_applied["y_ticks_bound_to_axis_for_explicit_point_chart"] = True

    if (
        axis_repair_hint.get("x_ticks_missing")
        and x_axis_type != NUMERIC_AXIS_TYPE
        and len(x_ticks_values) < len(x_pixel_positions)
    ):
        missing_count = len(x_pixel_positions) - len(x_ticks_values)
        x_ticks_values = list(x_ticks_values) + [f"category_{i + 1}" for i in range(missing_count)]

    if (
        axis_repair_hint.get("y_ticks_missing")
        and y_axis_type != NUMERIC_AXIS_TYPE
        and len(y_ticks_values) < len(y_pixel_positions)
    ):
        missing_count = len(y_pixel_positions) - len(y_ticks_values)
        y_ticks_values = [f"category_{i + 1}" for i in range(missing_count)] + list(y_ticks_values)

    if (
        normalize_axis_type_name(x_axis_type) == NUMERIC_AXIS_TYPE
        and normalize_axis_type_name(y_axis_type) == NUMERIC_AXIS_TYPE
    ):
        (
            repaired_x_axis,
            repaired_y_axis,
            repaired_x_ticks,
            repaired_y_ticks,
            repaired_x_pixels,
            repaired_y_pixels,
            normalized_prefix_repaired,
        ) = repair_normalized_point_prefix_axis(
            img,
            chart_type,
            x_axis,
            y_axis,
            x_ticks_values,
            y_ticks_values,
            x_pixel_positions,
            y_pixel_positions,
            axis_repair_hint,
        )
        if normalized_prefix_repaired:
            x_axis = repaired_x_axis
            y_axis = repaired_y_axis
            x_ticks_values = repaired_x_ticks
            y_ticks_values = repaired_y_ticks
            x_pixel_positions = repaired_x_pixels
            y_pixel_positions = repaired_y_pixels
            x_axis_scale = "linear"
            y_axis_scale = "linear"
            repair_applied["normalized_point_prefix_axis_repaired"] = True

    x_ticks_values, x_pixel_positions, x_placeholders_removed = _drop_mixed_placeholder_category_ticks(
        x_ticks_values,
        x_pixel_positions,
        chart_type,
        "x",
    )
    y_ticks_values, y_pixel_positions, y_placeholders_removed = _drop_mixed_placeholder_category_ticks(
        y_ticks_values,
        y_pixel_positions,
        chart_type,
        "y",
    )
    if x_placeholders_removed:
        repair_applied["x_category_placeholders_removed"] = True
    if y_placeholders_removed:
        repair_applied["y_category_placeholders_removed"] = True

    if not _is_bar_chart_type(chart_type):
        if normalize_axis_type_name(x_axis_type) == NUMERIC_AXIS_TYPE:
            x_pixel_positions, x_bound_scale, x_bound = bind_noisy_numeric_ticks_to_labels(
                img,
                merged_lines,
                x_axis,
                y_axis,
                chart_type,
                "x",
                x_axis,
                x_pixel_positions,
                x_ticks_values,
            )
            if x_bound:
                x_axis_scale = x_bound_scale
                repair_applied["x_ticks_bound_to_mllm_count"] = True
        if normalize_axis_type_name(y_axis_type) == NUMERIC_AXIS_TYPE:
            y_pixel_positions, y_bound_scale, y_bound = bind_noisy_numeric_ticks_to_labels(
                img,
                merged_lines,
                x_axis,
                y_axis,
                chart_type,
                "y",
                y_axis,
                y_pixel_positions,
                y_ticks_values,
            )
            if y_bound:
                y_axis_scale = y_bound_scale
                repair_applied["y_ticks_bound_to_mllm_count"] = True

        (
            repaired_x_axis,
            repaired_y_axis,
            repaired_x_pixels,
            repaired_y_pixels,
            repaired_x_scale,
            repaired_y_scale,
            point_span_repaired,
        ) = repair_point_tick_span_from_plot_bounds(
            img,
            chart_type,
            x_axis,
            y_axis,
            x_ticks_values,
            y_ticks_values,
            x_pixel_positions,
            y_pixel_positions,
            x_axis_scale,
            y_axis_scale,
            axis_repair_hint,
        )
        if point_span_repaired:
            x_axis = repaired_x_axis
            y_axis = repaired_y_axis
            x_pixel_positions = repaired_x_pixels
            y_pixel_positions = repaired_y_pixels
            x_axis_scale = repaired_x_scale
            y_axis_scale = repaired_y_scale
            repair_applied["point_tick_span_repaired_from_plot_bounds"] = True

    if _is_bar_chart_type(chart_type):
        if x_axis_type == NUMERIC_AXIS_TYPE:
            x_pixel_positions, x_bound = bind_noisy_numeric_bar_ticks_to_labels(
                chart_type,
                "x",
                x_axis,
                x_pixel_positions,
                x_ticks_values,
            )
            if x_bound:
                repair_applied["x_ticks_bound_to_label_count"] = True
        if y_axis_type == NUMERIC_AXIS_TYPE:
            y_pixel_positions, y_bound = bind_noisy_numeric_bar_ticks_to_labels(
                chart_type,
                "y",
                y_axis,
                y_pixel_positions,
                y_ticks_values,
            )
            if y_bound:
                repair_applied["y_ticks_bound_to_label_count"] = True
            elif len(y_ticks_values or []) < 2 and axis_repair_hint.get("y_ticks_missing") and len(y_pixel_positions) > 8:
                y_pixel_positions = _linspace_pixels_on_axis("y", y_axis, 6)
                repair_applied["y_ticks_capped_without_labels"] = True
        if x_axis_type == NUMERIC_AXIS_TYPE and len(x_ticks_values or []) < 2 and axis_repair_hint.get("x_ticks_missing") and len(x_pixel_positions) > 8:
            x_pixel_positions = _linspace_pixels_on_axis("x", x_axis, 6)
            repair_applied["x_ticks_capped_without_labels"] = True

        if chart_type in {"v_bar", "v_stacked_bar"} and y_axis_type == NUMERIC_AXIS_TYPE:
            (
                repaired_x_axis,
                repaired_y_axis,
                repaired_y_pixels,
                repaired_y_scale,
                y_axis_repaired,
            ) = repair_suspicious_vbar_value_axis(
                img,
                merged_lines,
                x_axis,
                y_axis,
                repair_boxes or _bar_boxes(img, chart_type),
                y_ticks_values,
                y_pixel_positions,
                axis_repair_hint,
            )
            if y_axis_repaired:
                x_axis = repaired_x_axis
                y_axis = repaired_y_axis
                y_pixel_positions = repaired_y_pixels
                y_axis_scale = repaired_y_scale
                repair_applied["y_axis_repaired_from_vbar_value_bounds"] = True
                repair_applied["y_ticks_repaired_from_vbar_value_bounds"] = True

    x_pixel_positions, x_snapped = snap_numeric_ticks_to_visual_grid(
        img,
        merged_lines,
        x_axis,
        y_axis,
        chart_type,
        "x",
        x_pixel_positions,
        x_ticks_values,
        x_axis_type,
    )
    y_pixel_positions, y_snapped = snap_numeric_ticks_to_visual_grid(
        img,
        merged_lines,
        x_axis,
        y_axis,
        chart_type,
        "y",
        y_pixel_positions,
        y_ticks_values,
        y_axis_type,
    )
    if x_snapped:
        repair_applied["x_ticks_snapped_to_visual_grid"] = True
    if y_snapped:
        repair_applied["y_ticks_snapped_to_visual_grid"] = True

    # Local fallback for offline/dev runs: when the LLM cannot return usable
    # tick labels, keep the image-processing pipeline alive by assigning
    # positional numeric ticks to every detected tick mark.
    if len(x_ticks_values) < 2 and len(x_pixel_positions) >= 2:
        logger.warning("LLM X tick labels unavailable; using positional fallback ticks.")
        x_ticks_values = list(range(len(x_pixel_positions)))
        x_axis_type = NUMERIC_AXIS_TYPE

    if len(y_ticks_values) < 2 and len(y_pixel_positions) >= 2:
        logger.warning("LLM Y tick labels unavailable; using positional fallback ticks.")
        y_ticks_values = list(range(len(y_pixel_positions)))
        y_axis_type = NUMERIC_AXIS_TYPE
    
    # 匹配X轴刻度
    if x_ticks_values and x_pixel_positions:
        # 确保数值数量与刻度数量匹配
        if len(x_ticks_values) > len(x_pixel_positions):
            x_ticks_values = x_ticks_values[:len(x_pixel_positions)]
        elif len(x_ticks_values) < len(x_pixel_positions):
            # 如果数值不足，使用已有数值插值
            if len(x_ticks_values) >= 2:
                import numpy as np
                # 尝试转换为数值进行插值
                try:
                    numeric_values = [float(v) for v in x_ticks_values]
                    x_positions = np.linspace(0, 1, len(x_pixel_positions))
                    orig_positions = np.linspace(0, 1, len(numeric_values))
                    interpolated = np.interp(x_positions, orig_positions, numeric_values)
                    x_ticks_values = interpolated.tolist()
                except:
                    # 如果是文字轴，无法插值，保持原样
                    pass
        
        # 匹配刻度值和像素位置
        for i, (tick_value, pixel_pos) in enumerate(zip(x_ticks_values, x_pixel_positions)):
            if x_axis_type == NUMERIC_AXIS_TYPE:
                try:
                    value = float(tick_value)
                    x_ticks_data.append(value)
                except ValueError:
                    # 如果是数值轴但值不是数字，转换为字符串
                    x_ticks_data.append(str(tick_value))
            else:
                # 文字轴，直接存储为字符串
                x_ticks_data.append(str(tick_value))
            x_pixels_data.append(pixel_pos)
    
    # 匹配Y轴刻度
    if y_ticks_values and y_pixel_positions:
        # 确保数值数量与刻度数量匹配
        if len(y_ticks_values) > len(y_pixel_positions):
            y_ticks_values = y_ticks_values[:len(y_pixel_positions)]
        elif len(y_ticks_values) < len(y_pixel_positions):
            # 如果数值不足，使用已有数值插值
            if len(y_ticks_values) >= 2:
                import numpy as np
                # 尝试转换为数值进行插值
                try:
                    numeric_values = [float(v) for v in y_ticks_values]
                    y_positions = np.linspace(0, 1, len(y_pixel_positions))
                    orig_positions = np.linspace(0, 1, len(numeric_values))
                    interpolated = np.interp(y_positions, orig_positions, numeric_values)
                    y_ticks_values = interpolated.tolist()
                except:
                    # 如果是文字轴，无法插值，保持原样
                    pass
        
        # 匹配刻度值和像素位置
        for i, (tick_value, pixel_pos) in enumerate(zip(y_ticks_values, y_pixel_positions)):
            if y_axis_type == NUMERIC_AXIS_TYPE:
                try:
                    value = float(tick_value)
                    y_ticks_data.append(value)
                except ValueError:
                    # 如果是数值轴但值不是数字，转换为字符串
                    y_ticks_data.append(str(tick_value))
            else:
                # 文字轴，直接存储为字符串
                y_ticks_data.append(str(tick_value))
            y_pixels_data.append(pixel_pos)
    
    if (
        len(x_ticks_data) < _required_tick_count(chart_type, "x")
        or len(y_ticks_data) < _required_tick_count(chart_type, "y")
    ):
        logger.warning(f"有效刻度数量不足: {image_path}")
        return None
    
    logger.debug(f"最终有效刻度: X轴={len(x_ticks_data)}个, Y轴={len(y_ticks_data)}个")
    
    # 阶段 9：生成加密 tick 和加密像素。只对数值轴插入中间 tick，文字轴保持原样。
    # 生成加密刻度和对应的加密像素位置
    logger.debug("生成加密刻度和对应的加密像素位置...")
    
    # 判断是否为数值轴
    is_x_numeric = normalize_axis_type_name(x_axis_type) == NUMERIC_AXIS_TYPE
    is_y_numeric = normalize_axis_type_name(y_axis_type) == NUMERIC_AXIS_TYPE
    if is_x_numeric:
        x_axis_scale = axis_scale_from_ticks_and_pixels(x_ticks_data, x_pixels_data)
    else:
        x_axis_scale = "linear"
    if is_y_numeric:
        y_axis_scale = axis_scale_from_ticks_and_pixels(y_ticks_data, y_pixels_data)
    else:
        y_axis_scale = "linear"

    if (
        chart_type in {"scatter", "bubble"}
        and is_x_numeric
        and is_y_numeric
        and _point_chart_grid_hint(axis_repair_hint)
        and (
            repair_applied.get("x_ticks_refined_from_projection_grid")
            or repair_applied.get("y_ticks_refined_from_projection_grid")
            or repair_applied.get("x_ticks_refined_from_missing_axis_grid")
            or repair_applied.get("y_ticks_refined_from_missing_axis_grid")
            or repair_applied.get("x_ticks_bootstrapped")
            or repair_applied.get("y_ticks_bootstrapped")
            or repair_applied.get("x_ticks_bootstrapped_after_filter")
            or repair_applied.get("y_ticks_bootstrapped_after_filter")
        )
        and not (
            repair_applied.get("axis_refined_from_projected_tick_grid")
            or repair_applied.get("axis_refined_from_missing_point_grid")
        )
    ):
        refined_x_axis, refined_y_axis, axes_refined = refine_point_chart_axes_from_projected_ticks(
            img,
            x_axis,
            y_axis,
            x_pixels_data,
            y_pixels_data,
            y_axis_scale,
            prefer_grid_bounds=bool(
                repair_applied.get("x_ticks_bootstrapped")
                or repair_applied.get("y_ticks_bootstrapped")
                or repair_applied.get("x_ticks_bootstrapped_after_filter")
                or repair_applied.get("y_ticks_bootstrapped_after_filter")
                or axis_repair_hint.get("plot_area_style") == "grid_only"
            ),
        )
        if axes_refined:
            x_axis, y_axis = refined_x_axis, refined_y_axis
            repair_applied["axis_refined_from_projected_tick_grid"] = True
            repair_applied["plot_bounds_refined_from_horizontal_grid"] = True
            if y_axis_scale != "linear":
                repair_applied["plot_vertical_bounds_refined_from_grid"] = True
            logger.debug(
                "Point-chart axes refined after projection tick binding: X=%s Y=%s y_scale=%s",
                x_axis,
                y_axis,
                y_axis_scale,
            )
    
    # 生成加密刻度（只对数字轴加密）
    x_ticks_encrypted = generate_encrypted_ticks(
        x_ticks_data,
        is_numeric_axis=is_x_numeric,
        axis_scale=x_axis_scale,
    )
    y_ticks_encrypted = generate_encrypted_ticks(
        y_ticks_data,
        is_numeric_axis=is_y_numeric,
        axis_scale=y_axis_scale,
    )
    
    # 生成对应的加密像素位置
    # 对于数字轴，需要插入中间像素位置；对于文字轴，不需要
    if is_x_numeric:
        x_pixels_encrypted = []
        for i in range(len(x_ticks_data)):
            # 添加原始像素位置
            x_pixels_encrypted.append(x_pixels_data[i])
            # 如果不是最后一个点，计算并添加中间像素位置
            if i < len(x_ticks_data) - 1:
                mid_pixel = (x_pixels_data[i] + x_pixels_data[i + 1]) // 2
                x_pixels_encrypted.append(mid_pixel)
    else:
        # 文字轴，不插入中间像素位置
        x_pixels_encrypted = x_pixels_data.copy()
    
    if is_y_numeric:
        y_pixels_encrypted = []
        for i in range(len(y_ticks_data)):
            # 添加原始像素位置
            y_pixels_encrypted.append(y_pixels_data[i])
            # 如果不是最后一个点，计算并添加中间像素位置
            if i < len(y_ticks_data) - 1:
                mid_pixel = (y_pixels_data[i] + y_pixels_data[i + 1]) // 2
                y_pixels_encrypted.append(mid_pixel)
    else:
        # 文字轴，不插入中间像素位置
        y_pixels_encrypted = y_pixels_data.copy()
    
    logger.debug(f"生成加密刻度: X轴={len(x_ticks_encrypted)}个, Y轴={len(y_ticks_encrypted)}个")
    logger.debug(f"生成加密像素位置: X轴={len(x_pixels_encrypted)}个, Y轴={len(y_pixels_encrypted)}个")
    
    # 阶段 10：输出基础网格图、加密网格图和 ticks JSON，供后续系统生成 JSON 与评估预测使用。
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成基础网格图像 (_grid)
    basic_grid_path = os.path.join(output_dir, f"{chart_id}_grid.png")
    enhanced_grid_reconstruction = _run_enhanced_cartesian_grid_reconstruction(
        image_path,
        output_dir,
        disable_cache=disable_cache,
    )
    enhanced_encryption = _encrypted_ticks_from_enhanced_bindings(
        enhanced_grid_reconstruction,
        x_axis_scale=x_axis_scale,
        y_axis_scale=y_axis_scale,
    )
    x_grid_bounds = None
    y_grid_bounds = None
    x_label_styles = None
    y_label_styles = None
    enhanced_x = None
    enhanced_y = None
    if enhanced_encryption:
        enhanced_x = enhanced_encryption.get("x")
        enhanced_y = enhanced_encryption.get("y")
        if enhanced_x:
            x_ticks_data = enhanced_x["native_ticks"]
            x_pixels_data = enhanced_x["native_pixels"]
            x_ticks_encrypted = enhanced_x["encrypted_ticks"]
            x_pixels_encrypted = enhanced_x["encrypted_pixels"]
            x_axis_type = enhanced_x["axis_type"]
            x_grid_bounds = enhanced_x.get("bounds")
            x_label_styles = enhanced_x.get("encrypted_label_styles")
        if enhanced_y:
            y_ticks_data = enhanced_y["native_ticks"]
            y_pixels_data = enhanced_y["native_pixels"]
            y_ticks_encrypted = enhanced_y["encrypted_ticks"]
            y_pixels_encrypted = enhanced_y["encrypted_pixels"]
            y_axis_type = enhanced_y["axis_type"]
            y_grid_bounds = enhanced_y.get("bounds")
            y_label_styles = enhanced_y.get("encrypted_label_styles")
        logger.debug(
            "Encrypted ticks derived from enhanced bindings: x_pixels=%s x_encrypted=%s y_pixels=%s y_encrypted=%s",
            x_pixels_data,
            x_pixels_encrypted,
            y_pixels_data,
            y_pixels_encrypted,
        )
    enhanced_basic_grid_source = _enhanced_basic_grid_visual_path(enhanced_grid_reconstruction)
    if enhanced_basic_grid_source is not None:
        try:
            enhanced_basic_grid_img = _read_image_path(enhanced_basic_grid_source)
            composed_basic_grid_img = _compose_image_with_grid_visual(img, enhanced_basic_grid_img)
            success, encoded_img = cv2.imencode('.png', composed_basic_grid_img)
            if success:
                encoded_img.tofile(basic_grid_path)
                logger.debug(
                    "Basic cartesian grid composed from enhanced reconstruction: %s -> %s",
                    enhanced_basic_grid_source,
                    basic_grid_path,
                )
            else:
                raise ValueError("cv2.imencode failed")
        except Exception as e:
            logger.error(f"Failed to copy enhanced basic grid image: {str(e)}")
            enhanced_basic_grid_source = None
    if enhanced_basic_grid_source is None:
        try:
            basic_grid_img = draw_basic_grid(img, x_pixels_data, y_pixels_data, x_axis, y_axis)
            # 保存基础网格图像
            try:
                success, encoded_img = cv2.imencode('.png', basic_grid_img)
                if success:
                    encoded_img.tofile(basic_grid_path)
                    logger.debug(f"基础网格图像已保存到: {basic_grid_path}")
                else:
                    logger.error(f"无法编码基础网格图像: {basic_grid_path}")
            except Exception as e:
                logger.error(f"保存基础网格图像时出错: {str(e)}")
        except Exception as e:
            logger.error(f"绘制基础网格时出错: {e}")
    
    # 生成加密网格图像 (_with_grid)
    try:
        standardized_basic_grid_img = _standard_native_grid_image(
            img,
            enhanced_x,
            enhanced_y,
            x_pixels_data,
            y_pixels_data,
            x_axis,
            y_axis,
            grid_line_color=GRID_LINE_COLOR,
        )
        success, encoded_img = cv2.imencode('.png', standardized_basic_grid_img)
        if success:
            encoded_img.tofile(basic_grid_path)
            enhanced_basic_grid_source = (
                "standard_native_grid_from_enhanced_bindings"
                if enhanced_encryption
                else "standard_native_grid_from_detected_ticks"
            )
            logger.debug("Basic cartesian grid saved with standard gray dashed style: %s", basic_grid_path)
    except Exception as e:
        logger.warning("Failed to standardize basic grid style: %s", e)

    encrypted_grid_path = os.path.join(output_dir, f"{chart_id}_with_grid.png")
    colored_grid_path = os.path.join(output_dir, f"{chart_id}_with_grid_color.png")
    try:
        base_grid_img_for_encryption = None
        if os.path.exists(basic_grid_path):
            try:
                encoded_base_grid = np.fromfile(basic_grid_path, dtype=np.uint8)
                decoded_base_grid = cv2.imdecode(encoded_base_grid, cv2.IMREAD_COLOR)
                if decoded_base_grid is not None and decoded_base_grid.shape[:2] == img.shape[:2]:
                    base_grid_img_for_encryption = decoded_base_grid
            except Exception as e:
                logger.warning(f"Failed to load basic grid as encrypted base image: {e}")

        # 使用加密像素位置和加密刻度绘制加密网格
        # 传递轴类型参数，只对数字轴加密部分添加文本
        encrypted_grid_img = draw_encrypted_grid(
            img, 
            x_pixels_encrypted, 
            y_pixels_encrypted, 
            x_ticks_encrypted, 
            y_ticks_encrypted, 
            x_axis, 
            y_axis,
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
            base_x_pixels=x_pixels_data,
            base_y_pixels=y_pixels_data,
            base_grid_img=base_grid_img_for_encryption,
            x_grid_bounds=x_grid_bounds,
            y_grid_bounds=y_grid_bounds,
            x_label_styles=x_label_styles,
            y_label_styles=y_label_styles,
        )
        colored_grid_img = draw_encrypted_grid(
            img,
            x_pixels_encrypted,
            y_pixels_encrypted,
            x_ticks_encrypted,
            y_ticks_encrypted,
            x_axis,
            y_axis,
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
            base_x_pixels=x_pixels_data,
            base_y_pixels=y_pixels_data,
            base_grid_img=base_grid_img_for_encryption,
            x_grid_bounds=x_grid_bounds,
            y_grid_bounds=y_grid_bounds,
            x_label_styles=x_label_styles,
            y_label_styles=y_label_styles,
            grid_line_color=GRID_LINE_REVIEW_COLOR,
        )
        # 保存加密网格图像
        try:
            success, encoded_img = cv2.imencode('.png', encrypted_grid_img)
            if success:
                encoded_img.tofile(encrypted_grid_path)
                logger.debug(f"加密网格图像已保存到: {encrypted_grid_path}")
                # 验证文件是否成功保存
                if os.path.exists(encrypted_grid_path):
                    logger.debug(f"加密网格文件大小: {os.path.getsize(encrypted_grid_path)} 字节")
                else:
                    logger.warning(f"加密网格图像文件未找到: {encrypted_grid_path}")
            else:
                logger.error(f"无法编码加密网格图像: {encrypted_grid_path}")
        except Exception as e:
            logger.error(f"保存加密网格图像时出错: {str(e)}")
    except Exception as e:
        logger.error(f"绘制加密网格时出错: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
    
    # 保存刻度信息（包含_with_grid相关数据）。
    # backend/main.py 会把这个 sidecar 合并到系统生成 JSON，评估预测阶段也依赖这些字段。
    tick_data = {
        "chart_id": chart_id,
        "x_ticks": x_ticks_data,
        "y_ticks": y_ticks_data,
        "x_pixels": x_pixels_data,
        "y_pixels": y_pixels_data,
        "x_ticks_encrypted": x_ticks_encrypted,
        "y_ticks_encrypted": y_ticks_encrypted,
        "x_pixels_encrypted": x_pixels_encrypted,
        "y_pixels_encrypted": y_pixels_encrypted,
        "x_axis_type": x_axis_type,
        "y_axis_type": y_axis_type,
        "x_axis_scale": x_axis_scale,
        "y_axis_scale": y_axis_scale,
        "x_label_orientation_angle": enhanced_x.get("stable_label_angle", 0.0) if enhanced_x else 0.0,
        "y_label_orientation_angle": enhanced_y.get("stable_label_angle", 0.0) if enhanced_y else 0.0,
        "x_skipped_encrypted_intervals": (
            enhanced_x.get("skipped_encrypted_intervals", []) if enhanced_x else []
        ),
        "y_skipped_encrypted_intervals": (
            enhanced_y.get("skipped_encrypted_intervals", []) if enhanced_y else []
        ),
        "x_dense_encrypted_interval_warnings": (
            enhanced_x.get("dense_interval_warnings", []) if enhanced_x else []
        ),
        "y_dense_encrypted_interval_warnings": (
            enhanced_y.get("dense_interval_warnings", []) if enhanced_y else []
        ),
        "x_axis_encryption_policy": enhanced_x.get("axis_encryption_policy") if enhanced_x else None,
        "y_axis_encryption_policy": enhanced_y.get("axis_encryption_policy") if enhanced_y else None,
        "image_path": image_path,
        "basic_grid_path": basic_grid_path,
        "encrypted_grid_path": encrypted_grid_path,
        "colored_grid_path": colored_grid_path,
        "encrypted_label_style_version": ENCRYPTED_LABEL_STYLE_VERSION,
        "generation_cache_disabled": bool(disable_cache),
        "colors": colors_data,
        "series_color": series_color_from_items(colors_data),
        "axis_repair": repair_applied,
    }

    tick_data["enhanced_grid_reconstruction"] = enhanced_grid_reconstruction
    tick_data["encrypted_tick_source"] = (
        enhanced_encryption.get("source") if enhanced_encryption else "legacy_tick_positions"
    )
    tick_data["basic_grid_source"] = (
        str(enhanced_basic_grid_source) if enhanced_basic_grid_source is not None else "legacy_tick_grid"
    )
    
    output_json_path = os.path.join(output_dir, f"{chart_id}_ticks.json")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(tick_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"刻度信息已保存到: {output_json_path}")
    except Exception as e:
        logger.error(f"保存JSON时出错: {e}")
    
    logger.info(f"处理完成: {chart_id}")
    return tick_data

def generate_encrypted_ticks(original_ticks, is_numeric_axis=True, axis_scale="linear"):
    # 数值轴加密策略：
    # - linear: 相邻 tick 插入算术中点。
    # - log: 相邻 tick 插入几何中点。
    # 文字轴直接返回原 tick，不生成中间类别。
    """
    根据原始刻度生成加密刻度（在原刻度之间插入中间值）
    对于数字轴，插入中间值；对于文字轴，不插入中间值
    """
    encrypted_ticks = []
    if not is_numeric_axis:
        # 文字轴，不加密，直接返回原值
        return original_ticks.copy()
    
    # 数字轴，插入中间值
    for i in range(len(original_ticks)):
        encrypted_ticks.append(original_ticks[i])
        if i < len(original_ticks) - 1:
            # 检查是否为数值类型
            try:
                val1 = float(original_ticks[i])
                val2 = float(original_ticks[i + 1])
                if axis_scale == "log" and val1 > 0 and val2 > 0:
                    mid_value = math.sqrt(val1 * val2)
                    if max(abs(val1), abs(val2)) >= 100:
                        mid_value = round(mid_value)
                    elif max(abs(val1), abs(val2)) >= 1:
                        mid_value = round(mid_value, 4)
                else:
                    mid_value = (val1 + val2) / 2
                # 消除中间值的浮点数误差
                mid_value = round(float(f"{mid_value:.12f}"), 10)
                encrypted_ticks.append(mid_value)
            except (ValueError, TypeError):
                # 如果不是数值，不插入中间值
                pass
    return encrypted_ticks


def batch_process_charts(input_dirs, output_base_dir):
    """
    批量处理指定目录下的所有图表
    """
    for input_dir in input_dirs:
        # 确保路径使用正确的编码
        input_dir = os.path.abspath(input_dir)
        if not os.path.exists(input_dir):
            logger.error(f"目录不存在: {input_dir}")
            continue
        
        # 获取图表类型（bubble或scatter）
        chart_type = os.path.basename(input_dir)
        output_dir = os.path.join(output_base_dir, f"{chart_type}_with_grid")
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"处理目录: {input_dir}, 输出目录: {output_dir}")
        
        # 获取所有PNG图像
        image_paths = glob(os.path.join(input_dir, "*.png"))
        # 过滤掉可能包含'grid'的文件名，避免重复处理
        image_paths = [p for p in image_paths if 'grid' not in os.path.basename(p).lower()]
        logger.info(f"找到 {len(image_paths)} 张 {chart_type} 图表")
        
        # 逐个处理图像
        success_count = 0
        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:  # 每处理10个文件输出一次进度
                logger.info(f"进度: {i+1}/{len(image_paths)}")
            
            try:
                result = process_chart(image_path, output_dir)
                if result:
                    success_count += 1
            except Exception as e:
                logger.error(f"处理文件时出错: {image_path}, 错误: {e}")
        
        logger.info(f"{chart_type} 处理完成: {success_count}/{len(image_paths)} 成功")
    
    return success_count


def main():
    logger.info("开始执行网格生成程序...")
    
    # 定义输入和输出路径，使用原始字符串避免转义问题
    charts_base_dir = r"D:\home work\Agent.paper\test demo\backend\Grid_generation\generated_charts\test1110"
    
    output_base_dir = r"D:\home work\Agent.paper\test demo\backend\Grid_generation\generated_charts_with_grid"
    
    # 处理所有文件模式
    test_mode = True  # 设置为True处理少量测试文件
    
    if test_mode:
        logger.info("进入测试模式，只处理少量文件...")
        
        # 选择几个测试文件
        charts_base_dir = r"\Users\98185\Desktop\Grid_generation\generated_charts\test1110"
        test_files = []
        
        # 直接查找test1113目录下的所有PNG文件
        if os.path.exists(charts_base_dir):
            png_files = [f for f in os.listdir(charts_base_dir) if f.endswith('.png') and 'grid' not in f.lower()]
            # 添加所有找到的PNG文件
            test_files = [os.path.join(charts_base_dir, f) for f in png_files]
        
        logger.info(f"找到测试文件: {test_files}")
        
        for test_file in test_files:
            if os.path.exists(test_file):
                logger.info(f"测试文件处理: {test_file}")
                chart_type = os.path.basename(os.path.dirname(test_file))
                test_output_dir = os.path.join(output_base_dir, f"{chart_type}_with_grid")
                os.makedirs(test_output_dir, exist_ok=True)
                result = process_chart(test_file, test_output_dir)
                logger.info(f"测试文件处理结果: {'成功' if result else '失败'}")
        
        logger.info("测试模式处理完成！请检查生成的图像是否包含加密刻度文本")
    else:
        # 检查输入目录是否存在
        if not os.path.exists(charts_base_dir):
            logger.error(f"图表基础目录不存在: {charts_base_dir}")
            return
        
        logger.info(f"图表基础目录: {charts_base_dir}")
        
        # 创建输出目录
        os.makedirs(output_base_dir, exist_ok=True)
        logger.info(f"输出基础目录: {output_base_dir}")
        
        # 获取所有子目录作为图表类型
        chart_types = []
        for item in os.listdir(charts_base_dir):
            item_path = os.path.join(charts_base_dir, item)
            if os.path.isdir(item_path):
                chart_types.append(item)
        
        logger.info(f"找到 {len(chart_types)} 种图表类型: {', '.join(chart_types)}")
        
        # 单独处理每种图表类型，以便分别统计成功数量
        total_success = 0
        
        for chart_type in chart_types:
            chart_dir = os.path.join(charts_base_dir, chart_type)
            logger.info(f"开始处理 {chart_type} 图表...")
            
            output_dir = os.path.join(output_base_dir, f"{chart_type}_with_grid")
            os.makedirs(output_dir, exist_ok=True)
            
            # 先测试一个文件
            try:
                chart_files = [f for f in os.listdir(chart_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and 'grid' not in f.lower()]
                if chart_files:
                    test_file = os.path.join(chart_dir, chart_files[0])
                    logger.info(f"测试处理 {chart_type} 文件: {test_file}")
                    test_result = process_chart(test_file, output_dir)
                    if test_result:
                        logger.info("测试文件处理成功！")
                    else:
                        logger.warning("测试文件处理失败，可能需要调整参数")
            except Exception as e:
                logger.error(f"测试 {chart_type} 处理时出错: {e}")
            
            # 批量处理当前图表类型
            chart_success = batch_process_charts([chart_dir], output_base_dir)
            logger.info(f"{chart_type} 处理完成，成功处理 {chart_success} 个文件")
            total_success += chart_success
        
        logger.info(f"所有图表处理完成！总计成功处理: {total_success} 个文件")


if __name__ == "__main__":
    logger.debug("网格生成脚本启动")
    main()
    logger.debug("网格生成脚本结束")
