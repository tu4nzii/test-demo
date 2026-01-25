"""
通过序号快速运行 function_calling 子模块。
用法：python utils/run_by_index.py 1
自动对应到 function_calling/xxx.py 并运行。
支持多模块（1-3 或 1,3,5）与额外参数传递。
"""

import sys
import subprocess

# 模块索引映射：编号 → 模块路径（不含 .py）
MODULE_INDEX = {
    1: 'axis.detect_lines',
    2: 'ticks.detect_ticks',
    3: 'ticks.filter_ticks',
    4: 'axis.merge_lines',
    5: 'axis.infer_axes',
    6: 'image.draw_grid_from_ticks',
    7: 'ticks.recognize_tick_labels',
    8: 'label.filter_ticks_with_labels',
    9: 'postprocess',
    10: 'function_calling.polar_coordinates_copy',
    11: 'ticks.correct_ticks',
    12: 'ticks.fit_ticks',
    13: 'center_find',
    14: 'test',
    15: 'ticks.extract_ticks_with_llm',
    16: 'label.recognize_tick_text_box'
}

def print_usage():
    print('用法: python utils/run_by_index.py <模块序号/范围/逗号分隔> [额外参数]')
    print('示例: python utils/run_by_index.py 1-5 --idx 0')
    print('模块列表:')
    for k in sorted(MODULE_INDEX):
        print(f'  {k}: {MODULE_INDEX[k]}')

def parse_indices(mod_arg: str):
    indices = set()
    if '-' in mod_arg:
        start, end = map(int, mod_arg.split('-'))
        indices.update(range(start, end + 1))
    elif ',' in mod_arg:
        for part in mod_arg.split(','):
            indices.add(int(part.strip()))
    else:
        indices.add(int(mod_arg))
    return sorted(indices)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mod_arg = sys.argv[1]
    extra_args = sys.argv[2:]

    try:
        selected = parse_indices(mod_arg)
    except:
        print("❌ 无法解析模块编号，请检查格式")
        print_usage()
        sys.exit(1)

    for idx in selected:
        if idx not in MODULE_INDEX:
            print(f'❌ 无效模块序号: {idx}')
            print_usage()
            sys.exit(1)

        module_path = MODULE_INDEX[idx]
        if not module_path.startswith('function_calling.'):
            module_path = f'function_calling.{module_path}'

        cmd = [sys.executable, '-m', module_path] + extra_args
        print(f'▶️ 执行模块 {idx}: {module_path}')
        subprocess.run(cmd)
