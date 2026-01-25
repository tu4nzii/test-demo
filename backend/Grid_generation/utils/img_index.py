# utils/img_index.py
# 提供图片序号与文件名的映射，便于命令行简写
from config import IMG_PATHS

IMG_INDEX = {i+1: (name, path) for i, (name, path) in enumerate(IMG_PATHS.items())}

# 用法示例：
# from utils.img_index import IMG_INDEX
# name, path = IMG_INDEX[3]  # 获取第3张图片的name和path
