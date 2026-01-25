# 图像读写函数（如 load_image, save_debug_image）
import os
import cv2

def load_image(path):
    """读取图像，返回彩色图像"""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"[Warning] 图像读取失败: {path}")
    return image

def save_image(image, out_path):
    """保存图像"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, image)

def process_images(input_folder='basic_images', output_folder='out'):
    """批量读取 input_folder 中所有图片并保存到 output_folder"""
    supported_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in supported_exts):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            image = load_image(input_path)
            if image is not None:
                save_image(image, output_path)
                print(f"[Info] 处理完成: {filename}")
            else:
                print(f"[Error] 无法读取文件: {filename}")

# 内置调试函数
def image_io_main():
    print("🚀 开始批量处理 basic_images 文件夹中的图片...")
    process_images("data/basic_images", "data/out")
    # 这是一个典型的路径问题：
    # 💡 你运行 python utils/image_io.py 时，当前工作路径是 E:\hdu\论文\AD\Agent。
    # input_folder='basic_images' 意味着代码会查找 E:\hdu\论文\AD\Agent\basic_images，
    # output_folder='out' 意味着保存到 E:\hdu\论文\AD\Agent\out。
    print("✅ 处理完成，输出到 out 文件夹。")

if __name__ == '__main__':
    image_io_main()