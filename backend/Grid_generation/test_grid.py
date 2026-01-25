import os
import sys
import logging
from grid_generation import process_chart

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    # 测试单个图表的处理
    # 请修改为你实际的图表路径
    # image_dir = r"e:\hdu\论文\AD\Agent\generated_charts\test1110"
    image_dir = r"D:\home work\Agent.paper\test demo\backend\Grid_generation\generated_charts\test1110"
    
    # 查找目录中的第一个PNG文件作为测试
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and 'grid' not in f.lower()]
    
    if not image_files:
        logger.error(f"在目录 {image_dir} 中未找到PNG图像文件")
        return
    
    test_image = os.path.join(image_dir, image_files[0])
    logger.info(f"使用测试图像: {test_image}")
    
    # 输出目录
    output_dir = r"D:\home work\Agent.paper\test demo\backend\Grid_generation\generated_charts_with_grid\test1110"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理图表
    logger.info("开始处理图表...")
    result = process_chart(test_image, output_dir)
    
    if result:
        logger.info(f"处理成功！基础网格路径: {result['basic_grid_path']}")
        logger.info(f"处理成功！加密网格路径: {result['encrypted_grid_path']}")
        logger.info(f"加密网格文件存在: {os.path.exists(result['encrypted_grid_path'])}")
        if os.path.exists(result['encrypted_grid_path']):
            logger.info(f"加密网格文件大小: {os.path.getsize(result['encrypted_grid_path'])} 字节")
        # 打印加密刻度和像素位置信息
        logger.info(f"X轴加密刻度数量: {len(result['x_ticks_encrypted'])}")
        logger.info(f"Y轴加密刻度数量: {len(result['y_ticks_encrypted'])}")
        logger.info(f"X轴加密像素位置数量: {len(result['x_pixels_encrypted'])}")
        logger.info(f"Y轴加密像素位置数量: {len(result['y_pixels_encrypted'])}")
    else:
        logger.error("处理失败！")

if __name__ == "__main__":
    main()