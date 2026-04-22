import os
import sys
import uuid
import json
import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import io

# 设置 UTF-8 编码以支持 Windows 系统
if sys.platform == 'win32':
    # Windows 系统设置控制台编码为 UTF-8
    import codecs
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加项目根目录到系统路径
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# 导入图表处理相关模块
from function_call.chart_processor import ChartProcessorFactory
from function_call.chart_type import ChartTypeDetector

# 创建 FastAPI 应用
app = FastAPI(title="图表智能分析 API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置目录路径
UPLOAD_DIR = backend_dir / "data" / "upload"
PROCESSED_DIR = backend_dir / "data" / "processed"
RESULTS_DIR = backend_dir / "data" / "results"
OUTPUT_DIR = backend_dir / "data" / "output"

# 确保目录存在
for directory in [UPLOAD_DIR, PROCESSED_DIR, RESULTS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 存储图表信息的临时字典（实际应用中应使用数据库）
charts_db = {}

def safe_encode_string(s: str) -> str:
    """安全编码字符串，移除 emoji 和特殊字符"""
    if not isinstance(s, str):
        s = str(s)
    # 移除 emoji 字符
    s = s.replace('❌', '[错误]').replace('✅', '[成功]').replace('⚠️', '[警告]')
    # 安全编码
    try:
        s = s.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    except:
        s = str(s).encode('ascii', errors='replace').decode('ascii', errors='replace')
    return s

@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {"message": "图表智能分析 API", "version": "1.0.0"}

@app.post("/api/upload/")
async def upload_files(
    file: UploadFile = File(..., description="图片文件"),
    json_data: UploadFile = File(..., description="JSON 数据文件")
):
    """
    上传图片和 JSON 数据文件
    
    返回:
    - chart_id: 图表唯一标识符
    - chart_type: 图表类型（rose/radar）
    - confidence: 检测置信度
    """
    try:
        # 生成唯一的图表 ID
        chart_id = str(uuid.uuid4())
        
        # 保存上传的文件
        image_path = UPLOAD_DIR / f"{chart_id}_image{Path(file.filename).suffix}"
        json_path = UPLOAD_DIR / f"{chart_id}_data.json"
        
        # 保存图片文件
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 保存 JSON 文件
        with open(json_path, "wb") as buffer:
            shutil.copyfileobj(json_data.file, buffer)
        
        # 检测图表类型
        detector = ChartTypeDetector()
        try:
            detection_result = detector.detect_chart_type(str(image_path))
        except Exception as detect_error:
            # 如果检测失败，使用默认值
            print(f"图表类型检测失败，使用默认值: {safe_encode_string(str(detect_error))}")
            detection_result = {
                "type": "rose",
                "confidence": 0.5,
                "error": safe_encode_string(str(detect_error))
            }
        
        chart_type = detection_result.get("type", "rose")
        confidence = detection_result.get("confidence", 0.5)
        
        # 存储图表信息
        charts_db[chart_id] = {
            "chart_id": chart_id,
            "chart_type": chart_type,
            "confidence": confidence,
            "image_path": str(image_path),
            "json_path": str(json_path),
            "processed": False,
            "evaluated": False
        }
        
        return {
            "chart_id": chart_id,
            "chart_type": chart_type,
            "confidence": confidence
        }
    
    except Exception as e:
        # 确保错误消息不包含特殊字符，避免编码问题
        try:
            error_msg = str(e)
            # 移除 emoji 字符和其他特殊 Unicode 字符
            error_msg = error_msg.replace('❌', '[错误]').replace('✅', '[成功]').replace('⚠️', '[警告]')
            # 确保可以安全编码
            error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except:
            error_msg = "上传过程中发生错误，请检查文件格式是否正确"
        raise HTTPException(status_code=500, detail=f"上传失败: {error_msg}")

@app.post("/api/process/")
async def process_chart(chart_id: str = Query(..., description="图表 ID")):
    """
    处理图表（加密处理）
    
    参数:
    - chart_id: 图表 ID（查询参数）
    
    返回:
    - encrypted_image_url: 加密后图片的 URL 路径
    """
    try:
        # 检查图表是否存在
        if chart_id not in charts_db:
            raise HTTPException(status_code=404, detail="图表不存在")
        
        chart_info = charts_db[chart_id]
        
        # 如果已经处理过，直接返回
        if chart_info.get("processed") and "encrypted_image_path" in chart_info:
            encrypted_path = chart_info["encrypted_image_path"]
            return {
                "encrypted_image_url": f"/api/images/{Path(encrypted_path).name}"
            }
        
        # 创建图表处理器
        processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
        
        # 确保输出目录存在（按图表类型分类）
        chart_type_output_dir = OUTPUT_DIR / chart_info["chart_type"]
        chart_type_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备 JSON 文件：process_single_image 期望 JSON 文件在图片所在目录，且文件名为 {图片名}.json
        image_path_obj = Path(chart_info["image_path"])
        expected_json_name = image_path_obj.stem + ".json"  # 例如: fe445f22-c328-4318-bd7d-0e66ad312827_image.json
        expected_json_path = image_path_obj.parent / expected_json_name
        
        # 如果期望的 JSON 文件不存在，从上传的 JSON 文件复制
        if not expected_json_path.exists():
            # 读取上传的 JSON 文件内容
            with open(chart_info["json_path"], 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            # 保存到期望的位置
            with open(expected_json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"已复制 JSON 文件到: {expected_json_path}")
        
        # 处理图片（加密）- 这会生成 JSON 文件在 output_dir 中
        encrypted_image_path = processor.encode_image(
            chart_info["image_path"],
            str(chart_type_output_dir)
        )
        
        if encrypted_image_path is None:
            raise HTTPException(status_code=500, detail="图片加密处理失败")
        
        # 查找坐标轴（生成轴线数据）
        try:
            axis_data = processor.find_axis(chart_info["image_path"])
            if axis_data:
                # 保存轴线数据
                axes_json_path = chart_type_output_dir / f"{chart_id}_axes.json"
                with open(axes_json_path, 'w', encoding='utf-8') as f:
                    json.dump(axis_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"警告: 查找坐标轴失败: {e}")
        
        # 确保生成的 JSON 文件包含评估所需的所有字段
        # 根据图片文件名确定 JSON 文件名
        image_stem = image_path_obj.stem
        generated_json_path = chart_type_output_dir / f"{image_stem}.json"
        
        if generated_json_path.exists():
            try:
                # 读取生成的 JSON 文件
                with open(generated_json_path, 'r', encoding='utf-8') as f:
                    generated_data = json.load(f)
                
                # 添加评估所需的字段
                if 'chart_id' not in generated_data:
                    generated_data['chart_id'] = chart_id
                if 'chart_type' not in generated_data:
                    generated_data['chart_type'] = chart_info["chart_type"]
                
                # 添加 image_paths（评估器需要，使用绝对路径）
                if 'image_paths' not in generated_data:
                    # 确保使用绝对路径
                    no_grid_path = str(Path(chart_info["image_path"]).absolute())
                    with_grid_path = str(Path(encrypted_image_path).absolute())
                    generated_data['image_paths'] = {
                        'no_grid': no_grid_path,
                        'with_grid': with_grid_path
                    }
                
                # 添加 data 字段（从原始 JSON 读取）
                if 'data' not in generated_data:
                    with open(chart_info["json_path"], 'r', encoding='utf-8') as f:
                        original_json = json.load(f)
                    if 'data_points' in original_json:
                        generated_data['data'] = original_json['data_points']
                    elif 'data' in original_json:
                        generated_data['data'] = original_json['data']
                
                # 保存更新后的 JSON
                with open(generated_json_path, 'w', encoding='utf-8') as f:
                    json.dump(generated_data, f, ensure_ascii=False, indent=2)
                print(f"已更新 JSON 文件，添加评估所需字段: {generated_json_path}")
            except Exception as e:
                print(f"警告: 更新 JSON 文件时出错: {e}")
        
        # 更新图表信息
        chart_info["processed"] = True
        chart_info["encrypted_image_path"] = encrypted_image_path
        chart_info["output_dir"] = str(chart_type_output_dir)
        
        # 返回加密图片的 URL
        encrypted_filename = Path(encrypted_image_path).name
        return {
            "encrypted_image_url": f"/api/images/{encrypted_filename}"
        }
    
    except Exception as e:
        import traceback
        try:
            error_msg = str(e)
            # 移除 emoji 字符
            error_msg = error_msg.replace('❌', '[错误]').replace('✅', '[成功]').replace('⚠️', '[警告]')
            # 安全编码
            error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            
            # 处理堆栈跟踪信息
            tb_str = traceback.format_exc()
            tb_str = tb_str.replace('❌', '[错误]').replace('✅', '[成功]').replace('⚠️', '[警告]')
            tb_str = tb_str.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            
            error_detail = f"处理失败: {error_msg}"
            # 只在开发环境包含详细堆栈信息
            if os.getenv('DEBUG', 'false').lower() == 'true':
                error_detail += f"\n{tb_str}"
        except:
            error_detail = "处理失败，请稍后重试"
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/evaluate/")
async def evaluate_chart(chart_id: str = Query(..., description="图表 ID")):
    """
    评估图表并返回结果 URL
    
    参数:
    - chart_id: 图表 ID（查询参数）
    
    返回:
    - results_url: 评估结果的 URL 路径
    """
    try:
        # 检查图表是否存在
        if chart_id not in charts_db:
            raise HTTPException(status_code=404, detail="图表不存在")
        
        chart_info = charts_db[chart_id]
        
        # 检查是否已处理
        if not chart_info.get("processed"):
            raise HTTPException(status_code=400, detail="请先处理图表")
        
        # 如果已经评估过，直接返回
        if chart_info.get("evaluated") and "evaluation_results_path" in chart_info:
            results_filename = Path(chart_info["evaluation_results_path"]).name
            return {
                "results_url": f"/api/results/{results_filename}"
            }
        
        # 创建图表处理器
        processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
        
        # 准备评估数据路径
        # encode_image 生成的 JSON 文件名是基于图片文件名的，例如: {chart_id}_image.json
        output_dir = chart_info.get("output_dir", str(OUTPUT_DIR / chart_info["chart_type"]))
        
        # 根据图片文件名确定 JSON 文件名
        image_path_obj = Path(chart_info["image_path"])
        image_stem = image_path_obj.stem  # 例如: fe445f22-c328-4318-bd7d-0e66ad312827_image
        eval_data_json_path = Path(output_dir) / f"{image_stem}.json"
        
        # 如果文件不存在，尝试使用 chart_id 作为文件名
        if not eval_data_json_path.exists():
            eval_data_json_path = Path(output_dir) / f"{chart_id}.json"
        
        # 如果还是不存在，尝试处理数据
        if not eval_data_json_path.exists():
            # 尝试处理数据（但这需要 JSON 文件已经存在）
            processed_data = processor.process_data(
                chart_id,
                chart_info["image_path"],
                chart_info["json_path"],
                output_dir
            )
            if processed_data is None:
                # 列出目录中的文件以便调试
                available_files = list(Path(output_dir).glob("*.json"))
                file_list = [f.name for f in available_files]
                raise HTTPException(
                    status_code=500, 
                    detail=f"无法生成评估数据。期望的 JSON 文件: {eval_data_json_path.name}，目录中的文件: {file_list}"
                )
        
        # 执行评估 - evaluate 方法需要 JSON 文件路径
        evaluation_results = processor.evaluate(str(eval_data_json_path))
        
        # 保存评估结果
        results_path = RESULTS_DIR / f"{chart_id}_evaluation.json"
        processor.save_evaluation_results(evaluation_results, str(results_path))
        
        # 更新图表信息
        chart_info["evaluated"] = True
        chart_info["evaluation_results_path"] = str(results_path)
        
        # 返回结果 URL
        results_filename = results_path.name
        return {
            "results_url": f"/api/results/{results_filename}"
        }
    
    except Exception as e:
        import traceback
        try:
            error_msg = str(e)
            # 移除 emoji 字符
            error_msg = error_msg.replace('❌', '[错误]').replace('✅', '[成功]').replace('⚠️', '[警告]')
            # 安全编码
            error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            
            # 处理堆栈跟踪信息
            tb_str = traceback.format_exc()
            tb_str = tb_str.replace('❌', '[错误]').replace('✅', '[成功]').replace('⚠️', '[警告]')
            tb_str = tb_str.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            
            error_detail = f"评估失败: {error_msg}"
            # 只在开发环境包含详细堆栈信息
            if os.getenv('DEBUG', 'false').lower() == 'true':
                error_detail += f"\n{tb_str}"
        except:
            error_detail = "评估失败，请稍后重试"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """获取处理后的图片"""
    # 在 processed 目录中查找
    image_path = PROCESSED_DIR / filename
    if image_path.exists():
        return FileResponse(str(image_path))
    
    # 在 output 目录的各个子目录中查找
    for chart_type_dir in OUTPUT_DIR.iterdir():
        if chart_type_dir.is_dir():
            image_path = chart_type_dir / filename
            if image_path.exists():
                return FileResponse(str(image_path))
    
    # 如果不在 processed 目录，尝试在 upload 目录查找
    image_path = UPLOAD_DIR / filename
    if image_path.exists():
        return FileResponse(str(image_path))
    
    raise HTTPException(status_code=404, detail="图片不存在")

@app.get("/api/results/{filename}")
async def get_results(filename: str):
    """获取评估结果"""
    results_path = RESULTS_DIR / filename
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return JSONResponse(content=results)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

