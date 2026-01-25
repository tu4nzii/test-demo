def main(idx=None):
    import os
    import json
  

    from config import IMG_PATHS, DEBUG_OUTPUT_DIRS
    from utils.image_io import load_image, save_image

    # 设置 index 范围或指定图像名
    image_names = list(IMG_PATHS.keys())
    if idx is not None:
        image_names = [image_names[idx]]

    # 读取所有 tick label 框
    label_path = "data/debug/recognize_tick_labels/all_tick_label_boxes.json"
    with open(label_path, 'r') as f:
        all_labels = json.load(f)

    for image_name in image_names:
        print(f"\n⏳ Processing: {image_name}")
        tick_path = f"data/debug/detect_ticks/tick_data/{image_name}.json"
        out_json = f"data/debug/tick_data/{image_name}.filtered.json"
        out_vis = f"data/debug/vis/{image_name}_filtered.jpg"

        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        os.makedirs(os.path.dirname(out_vis), exist_ok=True)

        # 加载 tick 数据
        with open(tick_path, 'r') as f:
            tick_data = json.load(f)

        label_boxes = all_labels.get(image_name, [])

        # 过滤 tick
        x_keep, x_removed = filter_ticks(tick_data["x_ticks"], label_boxes, direction='x')
        y_keep, y_removed = filter_ticks(tick_data["y_ticks"], label_boxes, direction='y')

        tick_data["x_ticks"] = x_keep
        tick_data["y_ticks"] = y_keep

        with open(out_json, 'w') as f:
            json.dump(tick_data, f, indent=2)

        # 读取图像并绘制
        img = load_image(IMG_PATHS[image_name])
        img = draw_ticks_and_labels(img, x_keep + y_keep, x_removed + y_removed, label_boxes)
        save_image(img, out_vis)

        print(f"✅ Saved tick data: {out_json}")
        print(f"🖼️  Saved visual image: {out_vis}")

# ========= 工具函数 =========

def overlaps(tick, box, direction='x', margin=3):
    tx1, ty1, tx2, ty2 = tick
    bx1, by1, bx2, by2 = box
    if direction == 'x':  # 垂直 tick
        return (bx1 - margin <= tx1 <= bx2 + margin) and (by1 - margin <= ty1 <= by2 + margin)
    else:  # 水平 tick
        return (by1 - margin <= ty1 <= by2 + margin) and (bx1 - margin <= tx1 <= bx2 + margin)

def filter_ticks(ticks, label_boxes, direction='x'):
    keep, remove = [], []
    for tick in ticks:
        if any(overlaps(tick, label['box'], direction) for label in label_boxes):
            remove.append(tick)
        else:
            keep.append(tick)
    return keep, remove

def draw_ticks_and_labels(image, kept_ticks, removed_ticks, label_boxes):
    for x1, y1, x2, y2 in kept_ticks:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # green
    for x1, y1, x2, y2 in removed_ticks:
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 1)  # magenta
    for label in label_boxes:
        x1, y1, x2, y2 = label["box"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)  # yellow
    return image

# ✅ 手动运行
if __name__ == "__main__":
    import cv2  
    main()
