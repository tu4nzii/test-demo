import os
import json
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Optional
import shutil
import tempfile
import fastapi_cdn_host
import uvicorn
# 自动切换到快速 CDN

# 导入现有的玫瑰图处理模块
from demo_rose.demo_rose_circle_find import RoseChartEncoder
from demo_rose.demo_axis_find_rose import RoseChartAxisFinder
from demo_rose.demo_data_solve_rose import process_rose_evaluation_data
from demo_rose.demo_evaluation_rose import RoseChartEvaluator
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI(title="图表分析后端服务", description="提供图表（如玫瑰图）图像处理、坐标轴识别、数据处理和评估功能")
fastapi_cdn_host.patch_docs(app)
# Add this CORS middleware section
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 配置路径
UPLOAD_DIR = "./data/upload"
OUTPUT_DIR = "./data/output/rose"
FEEDBACK_DIR = "./data/feedback"
AMPLIFIER_DIR = "./data/amplifier/rose"

# 创建必要的目录
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)
os.makedirs(AMPLIFIER_DIR, exist_ok=True)

@app.post("/api/upload/", response_model=Dict)
async def upload_chart(file: UploadFile = File(...), json_data: UploadFile = File(...)):
    """上传图表和相关数据"""
    try:
        # 1. 保存上传的图片
        json_content = json.loads(await json_data.read())
        file_ext = os.path.splitext(file.filename)[1] if os.path.splitext(file.filename)[1] else '.png'
        # print(json_content)
        chart_id = json_content.get('chart_id', f"rose_{int(time.time())}")
        image_path = os.path.join(UPLOAD_DIR, f"{chart_id}{file_ext}")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. 解析并保存JSON数据
        json_path = os.path.join(UPLOAD_DIR, f"{chart_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success",
            "chart_id": chart_id,
            "message": "文件上传成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.post("/api/process/", response_model=Dict)
async def process_chart(chart_id: str):
    """处理图表：加密、找轴、处理数据"""
    try:
        # 1. 准备文件路径
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_path = None
        
        for ext in image_extensions:
            potential_path = os.path.join(UPLOAD_DIR, f"{chart_id}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            raise HTTPException(status_code=404, detail=f"未找到ID为{chart_id}的图像文件")
        
        json_path = os.path.join(UPLOAD_DIR, f"{chart_id}.json")
        if not os.path.exists(json_path):
            raise HTTPException(status_code=404, detail=f"未找到ID为{chart_id}的JSON文件")
        
        # 2. 使用demo_rose_circle_find处理图片，获取加密后的图片
        encoder = RoseChartEncoder()
        encrypted_image_path = encoder.process_single_image(image_path, OUTPUT_DIR)
        
        if not encrypted_image_path:
            raise HTTPException(status_code=500, detail="图像加密处理失败")
        
        # 3. 使用demo_axis_find_rose找到坐标轴
        axis_finder = RoseChartAxisFinder()
        axis_result = axis_finder.process_single_image(image_path)
        
        if not axis_result:
            raise HTTPException(status_code=500, detail="坐标轴识别失败")
        
        # 4. 使用demo_data_solve_rose处理数据
        # 这里需要注意，process_rose_evaluation_data函数默认处理的是rose_001.json
        # 我们需要修改它以处理指定的chart_id
        # 先复制文件到预期的位置
        target_json_path = os.path.join(OUTPUT_DIR, f"{chart_id}.json")
        if os.path.exists(target_json_path):
            # 读取现有的结果文件
            with open(target_json_path, 'r', encoding='utf-8') as f:
                rose_data = json.load(f)
            
            # 添加必要的字段
            rose_data['chart_id'] = chart_id
            rose_data['chart_type'] = 'rose'
            rose_data['image_paths'] = {
                'no_grid': image_path,
                'with_grid': encrypted_image_path
            }
            
            # 转换labels和values为data键值对
            with open(json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            rose_data['data'] = {}
            if 'data_points' in original_data:
                for label, value in original_data['data_points'].items():
                    rose_data['data'][label] = value
            
            # 保存整合后的文件
            output_path = os.path.join(OUTPUT_DIR, 'result', f"{chart_id}_evalution_datasets.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(rose_data, f, ensure_ascii=False, indent=4)
        
        return {
            "status": "success",
            "chart_id": chart_id,
            "encrypted_image_url": f"/api/image/{chart_id}",
            "message": "图表处理成功"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/api/evaluate/", response_model=Dict)
async def evaluate_chart(chart_id: str):
    """评估处理后的图表"""
    try:
        # 1. 准备评估数据文件路径
        eval_data_path = os.path.join(OUTPUT_DIR, 'result',  f"{chart_id}_evalution_datasets.json")
        
        if not os.path.exists(eval_data_path):
            raise HTTPException(status_code=404, detail=f"未找到ID为{chart_id}的评估数据文件")
        
        # 2. 使用demo_evaluation_rose进行评估
        evaluator = RoseChartEvaluator()
        evaluator.process_single_image(eval_data_path)
        output_path = os.path.join(OUTPUT_DIR, 'result',  f"{chart_id}_evaluation_results.json")
        evaluator.save_results(output_path)
        
        return {
            "status": "success",
            "chart_id": chart_id,
            "results_url": f"/api/results/{chart_id}",
            "output_path": output_path,
            "message": "图表评估成功"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")

@app.get("/api/image/{chart_id}") # 【核心修改点 2】: 路径参数从 image_name 改为 chart_id
async def get_image(chart_id: str):
    """获取处理后的图像"""
    # 【核心修改点 3】: 在函数内部根据 chart_id 构造完整的文件名
    # 我们知道加密后的文件名固定为 {chart_id}_encode.png
    image_name = f"{chart_id}_encode.png"
    image_path = os.path.join(OUTPUT_DIR, image_name)
    
    print(f"正在根据 chart_id '{chart_id}' 尝试提供图片: {image_path}") # 增加日志方便调试
    
    if not os.path.exists(image_path):
        # 尝试查找其他可能的扩展名，增加鲁棒性
        image_name_jpg = f"{chart_id}_encode.jpg"
        image_path_jpg = os.path.join(OUTPUT_DIR, image_name_jpg)
        if os.path.exists(image_path_jpg):
            return FileResponse(image_path_jpg)
            
        raise HTTPException(status_code=404, detail=f"ID为 '{chart_id}' 的图像文件不存在")
        
    return FileResponse(image_path)

@app.get("/api/results/{chart_id}")
async def get_results(chart_id: str):
    """获取评估结果"""
    results_path = os.path.join(OUTPUT_DIR, 'result', f"{chart_id}_evaluation_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="评估结果不存在")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results

@app.get("/")
async def root():
    """根路由，返回API信息"""
    return {
        "api": "玫瑰图分析后端服务",
        "version": "1.0",
        "endpoints": [
            {"method": "POST", "path": "/api/upload/", "description": "上传玫瑰图和相关数据"},
            {"method": "POST", "path": "/api/process/", "description": "处理玫瑰图：加密、找轴、处理数据"},
            {"method": "POST", "path": "/api/evaluate/", "description": "评估处理后的玫瑰图"},
            {"method": "GET", "path": "/api/image/{chart_id}", "description": "获取处理后的图像"},
            {"method": "GET", "path": "/api/results/{chart_id}", "description": "获取评估结果"}
        ]
    }



if __name__ == "__main__":

    print("玫瑰图分析后端服务启动中...")
    print("访问 http://127.0.0.1:8000 查看API信息")
    print("访问 http://127.0.0.1:8000/docs 查看Swagger文档")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
