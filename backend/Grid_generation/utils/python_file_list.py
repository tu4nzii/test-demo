# python utils/python_file_list.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "python_file_list.txt")

# 默认需要遍历的目录
TARGET_DIRS = [
    os.path.join(PROJECT_ROOT),
    os.path.join(PROJECT_ROOT, "utils")
]

# 忽略的目录名（软编码）
IGNORE_DIR_NAMES = {".conda", ".venv", "__pycache__", ".git"}

def should_ignore(path):
    """判断路径中是否包含被忽略的目录"""
    parts = os.path.normpath(path).split(os.sep)
    return any(part in IGNORE_DIR_NAMES for part in parts)

def collect_py_files(directories, output_file):
    with open(output_file, 'w', encoding='utf-8') as out:
        for dir_path in directories:
            for root, _, files in os.walk(dir_path):
                if should_ignore(root):
                    continue
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, PROJECT_ROOT)
                        out.write(relative_path + '\n')

if __name__ == "__main__":
    collect_py_files(TARGET_DIRS, OUTPUT_FILE)
    print(f"✅ Python 文件路径列表已导出到：{OUTPUT_FILE}")
