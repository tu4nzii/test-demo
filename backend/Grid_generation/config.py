# config.py
import os

# # ===== 图像路径配置 =====
# IMG_PATHS = {
#     # 'graph_c1_no_grid_1.0': 'data/basic_images/rect/graph_c1_no_grid_1.0.png',
#     # 'graph_c2_no_grid_1.0': 'data/basic_images/rect/graph_c2_no_grid_1.0.png',
#     # 'graph_g1_no_grid_1.0': 'data/basic_images/rect/graph_g1_no_grid_1.0.png',
#     # 'graph_g2_no_grid_1.0': 'data/basic_images/rect/graph_g2_no_grid_1.0.png',
#     # 'graph_g3_no_grid_1.0': 'data/basic_images/rect/graph_g3_no_grid_1.0.png',
#     # 'graph1_no_grid_1.0': 'data/basic_images/rect/graph1_no_grid_1.0.png',
#     # 'graph2_no_grid_1.0': 'data/basic_images/rect/graph2_no_grid_1.0.png',
#     # 'graph3_no_grid_1.0': 'data/basic_images/rect/graph3_no_grid_1.0.png',
#     # 'graph4_no_grid_1.0': 'data/basic_images/rect/graph4_no_grid_1.0.png',
#     "bar_chart000": 'data/precision/bar/chart000.png',
#     "bar_chart001": 'data/precision/bar/chart001.png',
#     "bar_chart002": 'data/precision/bar/chart002.png',
#     "bar_chart003": 'data/precision/bar/chart003.png',
#     "bar_chart004": 'data/precision/bar/chart004.png',
#     "bubble_chart000": 'data/precision/bubble/chart000.png',
#     "bubble_chart001": 'data/precision/bubble/chart001.png',
#     "bubble_chart002": 'data/precision/bubble/chart002.png',
#     "bubble_chart003": 'data/precision/bubble/chart003.png',
#     "bubble_chart004": 'data/precision/bubble/chart004.png',
#     "line_chart000": 'data/precision/line/chart000.png',
#     "line_chart001": 'data/precision/line/chart001.png',
#     "line_chart002": 'data/precision/line/chart002.png',
#     "line_chart003": 'data/precision/line/chart003.png',
#     "line_chart004": 'data/precision/line/chart004.png',
#     "scatter_chart000": 'data/precision/scatter/chart000.png',
#     "scatter_chart001": 'data/precision/scatter/chart001.png',
#     "scatter_chart002": 'data/precision/scatter/chart002.png',
#     "scatter_chart003": 'data/precision/scatter/chart003.png',
#     "scatter_chart004": 'data/precision/scatter/chart004.png',
#     "stacked_bar_chart000": 'data/precision/stacked_bar/chart000.png',
#     "stacked_bar_chart001": 'data/precision/stacked_bar/chart001.png',
#     "stacked_bar_chart002": 'data/precision/stacked_bar/chart002.png',
#     "stacked_bar_chart003": 'data/precision/stacked_bar/chart003.png',
#     "stacked_bar_chart004": 'data/precision/stacked_bar/chart004.png'
#     # "chart4_no_grid": 'data/basic_images/rect/chart4_no_grid.png',
#     # 'chart5_no_grid': 'data/basic_images/rect/chart5_no_grid.png',
#     # 'graph6_no_grid': 'data/basic_images/rect/graph6_no_grid.png',
#     # 'graph8_no_grid': 'data/basic_images/rect/graph8_no_grid.png'
# }

# ===== 图像路径配置 =====
IMG_PATHS = {}

# 自动遍历 data/precision 下所有子目录
precision_root = './Grid_generation'
chart_types = os.listdir(precision_root)

for chart_type in chart_types:
    subdir = os.path.join(precision_root, chart_type)
    if not os.path.isdir(subdir):
        continue
    for fname in os.listdir(subdir):
        if fname.endswith(".png"):
            index = fname.replace(".png", "")  # chart000
            name = f"{chart_type}_chart{index[-3:]}"  # bar_chart000
            IMG_PATHS[name] = os.path.join(subdir, fname)



# ===== 输出路径配置 =====
DEBUG_OUTPUT_ROOT = 'data/debug/'
DEBUG_OUTPUT_DIRS = {
    'detect_lines': 'data/debug/detect_lines/',
    'detect_ticks': 'data/debug/detect_ticks/',
    'filter_ticks': 'data/debug/filter_ticks/',
    'merge_lines': 'data/debug/merge_lines/',
    'infer_axes': 'data/debug/infer_axes/',
    'draw_grid_from_ticks': 'data/debug/draw_grid_from_ticks/',
    'recognize_tick_labels': 'data/debug/recognize_tick_labels/',
    'filter_ticks_with_labels': 'data/debug/filter_ticks_with_labels/',
    'postprocess': 'data/debug/postprocess/',
    'polar_coordinates_copy': 'data/debug/polar_coordinates_copy/',
    'correct_ticks': 'data/debug/correct_ticks/',
    'fit_ticks': 'data/debug/fit_ticks/',
    'center_find': 'data/debug/center_find/',
    'test': 'data/debug/test/',
    'extract_ticks_with_llm': 'data/debug/extract_ticks_with_llm/'
}

# ===== Gemini 大模型配置 =====
# ✅ 推荐在部署或生产环境用环境变量覆盖
GEMINI_API_KEY = "AIzaSyA7nqKYMNvwa38xinXDb9E3WIPhn_eeoVI"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# ===== 输出是否自动清空 =====
CLEAR_OUTPUT_BEFORE_RUN = {
    'detect_lines': True,
    'detect_ticks': True,
    'filter_ticks': True,
    'merge_lines': True,
    'infer_axes': True,
    'draw_grid_from_ticks': True,
    'recognize_tick_labels': True,
    'filter_ticks_with_labels': True,
    'postprocess': True,
    'polar_coordinates_copy': True,
    'correct_ticks': True,
    'fit_ticks': True,
    'center_find': True,
    'test': True,
    'extract_ticks_with_llm': True  # ✅ 新增模块
}

# ===== 网格绘制参数 =====
GRID_CONFIG = {
    'grid_color': (180, 180, 180),
    'grid_thickness': 1,
    'grid_line_type': 16,  # cv2.LINE_AA
    'show_result': False
}

