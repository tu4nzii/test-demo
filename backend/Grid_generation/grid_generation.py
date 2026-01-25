import os
import cv2
import numpy as np
import json
import logging
import sys
import asyncio
import aiohttp
import base64
from glob import glob

# 配置日志系统
# 使用原始字符串避免路径问题
log_path = r"D:\home work\Agent.paper\test demo\backend\Grid_generation\grid_generation.log"
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

# ========== 大模型API配置项 ==========
api_key = "sk-f08TVXG8bJyobMmFvOgh09Bn93vFiuRX8j5iNuSSYQLmqgBd"
url = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

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

    # 构造请求体
    payload = {
        "model": "internvl3-78b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0
    }

    # 发送请求
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            try:
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
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
from function_calling.axis.detect_lines import detect_candidate_lines
from function_calling.axis.merge_lines import merge_similar_lines
from function_calling.axis.infer_axes import infer_axes_from_lines
from function_calling.ticks.detect_ticks import scan_pixels_for_ticks
from function_calling.ticks.filter_ticks import filter_ticks
from function_calling.label.recognize_tick_labels import recognize_tick_labels
from function_calling.image.draw_grid_from_ticks import draw_grid_from_ticks
from utils.image_io import load_image, save_image


def draw_basic_grid(img, x_pixels, y_pixels, x_axis, y_axis):
    """
    绘制基础网格 - 只延伸短横线形成网格图
    """
    canvas = img.copy()
    # 绘制坐标轴
    cv2.line(canvas, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
    cv2.line(canvas, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (255, 0, 0), 2)
    
    # 绘制水平网格线（Y方向）
    x_min, x_max = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    y_min, y_max = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])
    
    # 绘制垂直网格线
    for x_pix in x_pixels:
        cv2.line(canvas, (x_pix, y_min), (x_pix, y_max), (180, 180, 180), 1, cv2.LINE_AA)
    
    # 绘制水平网格线
    for y_pix in y_pixels:
        cv2.line(canvas, (x_min, y_pix), (x_max, y_pix), (180, 180, 180), 1, cv2.LINE_AA)
    
    return canvas

def draw_encrypted_grid(img, x_pixels, y_pixels, x_ticks_encrypted, y_ticks_encrypted, x_axis, y_axis):
    """
    绘制加密网格 - 在基础网格上添加加密刻度和文本
    """
    # 先绘制基础网格
    canvas = draw_basic_grid(img, x_pixels, y_pixels, x_axis, y_axis)
    
    # 绘制加密刻度文本标签，优化显示效果
    try:
        # 优化文本样式，减小字体大小避免重叠
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3  # 减小字体大小
        font_color = (0, 0, 255)  # 红色文本
        thickness = 1  # 细线
        padding = 3  # 减小内边距
        
        logger.debug(f"准备绘制加密刻度文本: X轴像素点数量={len(x_pixels)}, 加密刻度数量={len(x_ticks_encrypted)}")
        logger.debug(f"Y轴像素点数量={len(y_pixels)}, 加密刻度数量={len(y_ticks_encrypted)}")
        
        # 为X轴绘制加密刻度文本
        drawn_x_texts = 0
        x_min, x_max = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
        x_axis_y = max(y_axis[1], y_axis[3])  # X轴的Y坐标
        
        # 确保x_pixels和x_ticks_encrypted长度匹配
        if len(x_pixels) == len(x_ticks_encrypted):
            # 智能过滤：避免标签太密集导致重叠
            skip_factor = max(1, len(x_pixels) // 10)  # 根据点数量决定跳过因子
            
            for i in range(0, len(x_pixels), skip_factor):
                x_pix = x_pixels[i]
                # 检查索引是否有效
                if i < len(x_ticks_encrypted):
                    tick_value = x_ticks_encrypted[i]
                    
                    # 确保值有效并格式化，限制小数位数
                    if tick_value is not None:
                        text = f"{tick_value:.1f}" if isinstance(tick_value, float) else str(tick_value)
                        
                        # 获取文本大小
                        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                        
                        # 计算文本位置 - 放在X轴上方，确保足够空间
                        text_x = x_pix - text_size[0] // 2
                        # 确保文本在X轴上方有足够空间，避免重叠
                        text_y = x_axis_y - 8  # 适当间距
                        
                        # 边界检查，确保不与图表内容重叠
                        chart_content_margin = 50  # 图表内容边缘距离
                        if (0 <= text_x and text_x + text_size[0] <= canvas.shape[1] and \
                           0 <= text_y - text_size[1] - padding and text_y + padding <= canvas.shape[0] and \
                           text_y >= chart_content_margin):  # 确保在图表内容下方
                            # 使用半透明背景，减少对图表的遮挡
                            overlay = canvas.copy()
                            cv2.rectangle(overlay, 
                                        (text_x - padding, text_y - text_size[1] - padding),
                                        (text_x + text_size[0] + padding, text_y + padding),
                                        (255, 255, 255), -1)
                            # 添加透明度
                            alpha = 0.7  # 透明度因子
                            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
                            # 添加细边框
                            cv2.rectangle(canvas, 
                                        (text_x - padding, text_y - text_size[1] - padding),
                                        (text_x + text_size[0] + padding, text_y + padding),
                                        (0, 0, 0), 1)
                            # 绘制红色文本
                            cv2.putText(canvas, text, (text_x, text_y), 
                                        font, font_scale, font_color, thickness, cv2.LINE_AA)
                            drawn_x_texts += 1
        
        # 为Y轴绘制加密刻度文本
        drawn_y_texts = 0
        y_min, y_max = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])
        y_axis_x = min(x_axis[0], x_axis[2])  # Y轴的X坐标
        
        # 确保y_pixels和y_ticks_encrypted长度匹配
        if len(y_pixels) == len(y_ticks_encrypted):
            # 智能过滤：避免标签太密集导致重叠
            skip_factor = max(1, len(y_pixels) // 10)  # 根据点数量决定跳过因子
            
            for i in range(0, len(y_pixels), skip_factor):
                y_pix = y_pixels[i]
                # 检查索引是否有效
                if i < len(y_ticks_encrypted):
                    tick_value = y_ticks_encrypted[i]
                    
                    # 确保值有效并格式化，限制小数位数
                    if tick_value is not None:
                        text = f"{tick_value:.1f}" if isinstance(tick_value, float) else str(tick_value)
                        
                        # 获取文本大小
                        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                        
                        # 计算文本位置 - 放在Y轴左侧，避免与图表重叠
                        text_x = y_axis_x - text_size[0] - 8  # 放在Y轴左侧
                        text_y = y_pix + text_size[1] // 2
                        
                        # 边界检查，确保不与图表内容重叠
                        chart_content_margin = 50  # 图表内容边缘距离
                        if (0 <= text_y and text_y - text_size[1] - padding >= 0 and \
                           text_x - padding >= 0 and text_x + text_size[0] + padding <= canvas.shape[1] and \
                           text_x <= canvas.shape[1] - chart_content_margin):  # 确保在图表内容左侧
                            # 使用半透明背景，减少对图表的遮挡
                            overlay = canvas.copy()
                            cv2.rectangle(overlay, 
                                        (text_x - padding, text_y - text_size[1] - padding),
                                        (text_x + text_size[0] + padding, text_y + padding),
                                        (255, 255, 255), -1)
                            # 添加透明度
                            alpha = 0.7  # 透明度因子
                            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
                            # 添加细边框
                            cv2.rectangle(canvas, 
                                        (text_x - padding, text_y - text_size[1] - padding),
                                        (text_x + text_size[0] + padding, text_y + padding),
                                        (0, 0, 0), 1)
                            # 绘制红色文本
                            cv2.putText(canvas, text, (text_x, text_y), 
                                        font, font_scale, font_color, thickness, cv2.LINE_AA)
                            drawn_y_texts += 1
        
        logger.debug(f"成功绘制加密刻度文本: X轴{drawn_x_texts}个, Y轴{drawn_y_texts}个")
        
        # 不再添加水印，避免干扰图表
                    
    except Exception as e:
        logger.error(f"绘制加密刻度文本时出错: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
    
    return canvas

def process_chart(image_path, output_dir):
    """
    处理单个图表，生成两种网格图像和刻度信息
    1. _grid: 基础网格 - 短横线延伸形成网格图
    2. _with_grid: 加密网格 - 在基础网格上添加加密刻度和文本
    """
    logger.info(f"处理图像: {image_path}")
    logger.debug(f"输出目录: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
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
        logger.debug(f"图像尺寸: {w}x{h}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
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
        
        # 2. 合并相似直线
        merged_lines = merge_similar_lines(raw_lines)
        logger.debug(f"合并后得到 {len(merged_lines)} 条直线")
        
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
        
        if x_axis is None or y_axis is None:
            logger.warning(f"未检测到 X/Y 轴: {image_path}")
            return None
        
        logger.debug(f"检测到坐标轴: X轴={x_axis}, Y轴={y_axis}")
        
        # 4. 检测刻度线
        logger.debug("开始检测刻度线...")
        x_raw_ticks = scan_pixels_for_ticks(img, x_axis, direction='x', scan_range=20)
        y_raw_ticks = scan_pixels_for_ticks(img, y_axis, direction='y', scan_range=20)
        logger.debug(f"检测到 X轴刻度 {len(x_raw_ticks)} 个, Y轴刻度 {len(y_raw_ticks)} 个")
        
        if not x_raw_ticks or not y_raw_ticks:
            logger.warning(f"未检测到足够的刻度线: {image_path}")
            return None
        
        # 5. 合并和过滤刻度线
        x_merged_ticks = merge_similar_lines(x_raw_ticks, angle_threshold=np.deg2rad(10))
        y_merged_ticks = merge_similar_lines(y_raw_ticks, angle_threshold=np.deg2rad(10))
        
        x_filtered_ticks = filter_ticks(x_merged_ticks, direction='x')
        y_filtered_ticks = filter_ticks(y_merged_ticks, direction='y')
        
        logger.debug(f"过滤后: X轴刻度 {len(x_filtered_ticks)} 个, Y轴刻度 {len(y_filtered_ticks)} 个")
        
        if not x_filtered_ticks or not y_filtered_ticks:
            logger.warning(f"未检测到有效的刻度线: {image_path}")
            return None
    
    except Exception as e:
        logger.error(f"前序处理出错: {e}")
        return None
    
    # 6. 识别刻度标签
    logger.debug("开始识别刻度标签...")
    x_labels = []
    y_labels = []
    
    try:
        # 使用大模型识别刻度标签
        logger.debug("使用大模型识别刻度标签...")
        x_labels = recognize_tick_labels_with_llm(img, x_filtered_ticks, direction='x')
        y_labels = recognize_tick_labels_with_llm(img, y_filtered_ticks, direction='y')
        
        # 如果大模型识别失败，回退到OCR方法
        if len(x_labels) < 3:
            logger.debug("大模型X轴识别不足，回退到OCR方法")
            for region_size in [25, 30, 35]:
                x_labels = recognize_tick_labels(img, x_filtered_ticks, direction='x', label_region=region_size)
                if len(x_labels) >= 3:
                    break
        
        if len(y_labels) < 3:
            logger.debug("大模型Y轴识别不足，回退到OCR方法")
            for region_size in [25, 30, 35]:
                y_labels = recognize_tick_labels(img, y_filtered_ticks, direction='y', label_region=region_size)
                if len(y_labels) >= 3:
                    break
    except Exception as e:
        logger.error(f"识别刻度标签时出错: {e}")
    
    logger.debug(f"识别到 X轴标签 {len(x_labels)} 个, Y轴标签 {len(y_labels)} 个")
    
    # 初始化刻度数据列表
    chart_id = os.path.splitext(os.path.basename(image_path))[0]  # 定义chart_id变量
    x_ticks_data = []
    x_pixels_data = []
    y_ticks_data = []
    y_pixels_data = []
    
    # 使用OCR识别结果生成刻度数据
    logger.debug("使用OCR识别结果生成刻度数据")
    if x_labels:
        for item in sorted(x_labels, key=lambda x: (x['tick'][0] + x['tick'][2]) // 2):
            try:
                value = float(item['text'])
                x_ticks_data.append(value)
                x_pixels_data.append((item['tick'][0] + item['tick'][2]) // 2)
            except ValueError:
                continue
    
    if y_labels:
        for item in sorted(y_labels, key=lambda x: (x['tick'][1] + x['tick'][3]) // 2, reverse=True):
            try:
                value = float(item['text'])
                y_ticks_data.append(value)
                y_pixels_data.append((item['tick'][1] + item['tick'][3]) // 2)
            except ValueError:
                continue
    
    # 如果OCR识别的刻度不足，尝试使用刻度线位置推断值
    if len(x_ticks_data) < 2 and x_filtered_ticks:
        logger.debug("OCR识别失败，尝试使用刻度线位置推断值")
        # 基于像素位置生成模拟值
        x_pixel_positions = sorted([(t[0] + t[2]) // 2 for t in x_filtered_ticks])
        min_pixel = min(x_pixel_positions)
        max_pixel = max(x_pixel_positions)
        if min_pixel != max_pixel:
            # 生成均匀分布的值
            range_size = max_pixel - min_pixel
            for i, pix in enumerate(x_pixel_positions):
                # 从0开始，按位置比例生成值
                value = (pix - min_pixel) / range_size * 100  # 生成0-100的值
                x_ticks_data.append(value)
                x_pixels_data.append(pix)
    
    if len(y_ticks_data) < 2 and y_filtered_ticks:
        # 基于像素位置生成模拟值
        y_pixel_positions = sorted([(t[1] + t[3]) // 2 for t in y_filtered_ticks])
        min_pixel = min(y_pixel_positions)
        max_pixel = max(y_pixel_positions)
        if min_pixel != max_pixel:
            # 生成均匀分布的值（Y轴通常是倒序的，从上到下增大）
            range_size = max_pixel - min_pixel
            for i, pix in enumerate(y_pixel_positions):
                # 从0开始，按位置比例生成值（注意Y轴方向）
                value = (max_pixel - pix) / range_size * 100  # 生成0-100的值
                y_ticks_data.append(value)
                y_pixels_data.append(pix)
    
    if len(x_ticks_data) < 2 or len(y_ticks_data) < 2:
        logger.warning(f"有效刻度数量不足: {image_path}")
        return None
    
    logger.debug(f"最终有效刻度: X轴={len(x_ticks_data)}个, Y轴={len(y_ticks_data)}个")
    
    # 生成加密刻度和对应的加密像素位置
    logger.debug("生成加密刻度和对应的加密像素位置...")
    
    # 生成加密刻度
    x_ticks_encrypted = generate_encrypted_ticks(x_ticks_data)
    y_ticks_encrypted = generate_encrypted_ticks(y_ticks_data)
    
    # 生成对应的加密像素位置
    x_pixels_encrypted = []
    y_pixels_encrypted = []
    
    # 为X轴生成加密像素位置
    for i in range(len(x_ticks_data)):
        # 添加原始像素位置
        x_pixels_encrypted.append(x_pixels_data[i])
        # 如果不是最后一个点，计算并添加中间像素位置
        if i < len(x_ticks_data) - 1:
            mid_pixel = (x_pixels_data[i] + x_pixels_data[i + 1]) // 2
            x_pixels_encrypted.append(mid_pixel)
    
    # 为Y轴生成加密像素位置
    for i in range(len(y_ticks_data)):
        # 添加原始像素位置
        y_pixels_encrypted.append(y_pixels_data[i])
        # 如果不是最后一个点，计算并添加中间像素位置
        if i < len(y_ticks_data) - 1:
            mid_pixel = (y_pixels_data[i] + y_pixels_data[i + 1]) // 2
            y_pixels_encrypted.append(mid_pixel)
    
    logger.debug(f"生成加密刻度: X轴={len(x_ticks_encrypted)}个, Y轴={len(y_ticks_encrypted)}个")
    logger.debug(f"生成加密像素位置: X轴={len(x_pixels_encrypted)}个, Y轴={len(y_pixels_encrypted)}个")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成基础网格图像 (_grid)
    basic_grid_path = os.path.join(output_dir, f"{chart_id}_grid.png")
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
    encrypted_grid_path = os.path.join(output_dir, f"{chart_id}_with_grid.png")
    try:
        # 使用加密像素位置和加密刻度绘制加密网格
        encrypted_grid_img = draw_encrypted_grid(img, x_pixels_encrypted, y_pixels_encrypted, x_ticks_encrypted, y_ticks_encrypted, x_axis, y_axis)
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
    
    # 保存刻度信息（包含_with_grid相关数据）
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
        "image_path": image_path,
        "basic_grid_path": basic_grid_path,
        "encrypted_grid_path": encrypted_grid_path
    }
    
    output_json_path = os.path.join(output_dir, f"{chart_id}_ticks.json")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(tick_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"刻度信息已保存到: {output_json_path}")
    except Exception as e:
        logger.error(f"保存JSON时出错: {e}")
    
    logger.info(f"处理完成: {chart_id}")
    return tick_data

def generate_encrypted_ticks(original_ticks):
    """
    根据原始刻度生成加密刻度（在原刻度之间插入中间值）
    """
    encrypted_ticks = []
    for i in range(len(original_ticks)):
        encrypted_ticks.append(original_ticks[i])
        if i < len(original_ticks) - 1:
            # 在两个刻度之间插入中间值
            mid_value = (original_ticks[i] + original_ticks[i + 1]) / 2
            encrypted_ticks.append(mid_value)
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