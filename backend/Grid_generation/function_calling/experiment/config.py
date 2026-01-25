import os
import json

# 每种实验配置：(prompt_type, image_type)
EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "with_grid"),
    ("feedback", "with_grid"),
    ("feedback_cropped", "with_grid"),
    ("feedback_crop_from_feedback", "with_grid"),
    ("feedback_crop_adaptive", "with_grid"),
]

REPEAT_TIMES = 3
MAX_ATTEMPTS = 10  # 每个点最多尝试10次来获得足够成功结果

def load_chart_configs(config_dir=".../data/basic_images/chart_configs/rect"):
    """
    读取 chart_configs 文件夹下所有 JSON 图表配置。
    每个 JSON 文件包含：
    {
      "chart_id": "...",
      "image_paths": {"with_grid": ..., "no_grid": ...},
      "x_ticks": [...], "y_ticks": [...],
      "x_pixels": [...], "y_pixels": [...],
      "data_points": {label: [x, y], ...}
    }
    """
    chart_configs = []
    for filename in os.listdir(config_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(config_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                config = json.load(f)
                chart_configs.append(config)
    return chart_configs

# 加载所有配置
DATASET_CONFIGS = load_chart_configs()
