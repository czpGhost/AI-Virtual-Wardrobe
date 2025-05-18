import numpy as np
from PIL import Image
import io
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from typing import Optional, Literal
import base64
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from typing import Optional,Literal

from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import (resize_and_center, list_dir, 
                             get_agnostic_mask_hd, get_agnostic_mask_dc, 
                             preprocess_garment_image)
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")

app = FastAPI(
    title="Leffa API",
    description="Controllable Person Image Generation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.mount("/examples", StaticFiles(directory="./ckpts/examples"), name="examples")

def numpy_to_bytes(image_array: np.ndarray, format: str = 'PNG') -> bytes:
    """将numpy数组转换为字节流"""
    img = Image.fromarray(image_array)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

class LeffaPredictor(object):
    def __init__(self):
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

        vt_model_dc = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon_dc.pth",
            dtype="float16",
        )
        self.vt_inference_dc = LeffaInference(model=vt_model_dc)

        pt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
            pretrained_model="./ckpts/pose_transfer.pth",
            dtype="float16",
        )
        self.pt_inference = LeffaInference(model=pt_model)

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type: Literal["viton_hd", "dress_code"] = "viton_hd",      # 添加类型约束
        vt_garment_type: Literal["upper_body", "lower_body", "dresses"] = "upper_body",  # 添加类型约束
        vt_repaint=False,
        preprocess_garment=False
    ):
        # 打开并resize源图片
        src_image = Image.open(src_image_path)
        src_image = resize_and_center(src_image, 768, 1024)

        # 虚拟试衣时可选预处理服装图
        if control_type == "virtual_tryon" and preprocess_garment:
            if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                ref_image = preprocess_garment_image(ref_image_path)
            else:
                raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
        else:
            ref_image = Image.open(ref_image_path)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        if control_type == "virtual_tryon":
            src_image = src_image.convert("RGB")
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))
            if vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
            elif vt_model_type == "dress_code":
                mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024))
        elif control_type == "pose_transfer":
            mask = Image.fromarray(np.ones_like(src_image_array) * 255)

        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
            elif vt_model_type == "dress_code":
                src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
                src_image_seg = Image.fromarray(src_image_seg_array)
                densepose = src_image_seg
        elif control_type == "pose_transfer":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
            src_image_iuv = Image.fromarray(src_image_iuv_array)
            densepose = src_image_iuv

        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        if control_type == "virtual_tryon":
            if vt_model_type == "viton_hd":
                inference = self.vt_inference_hd
            elif vt_model_type == "dress_code":
                inference = self.vt_inference_dc
        elif control_type == "pose_transfer":
            inference = self.pt_inference
        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,
        )
        gen_image = output["generated_image"][0]
        return np.array(gen_image), np.array(mask), np.array(densepose)

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint, preprocess_garment):
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "virtual_tryon",
            ref_acceleration,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            vt_repaint,
            preprocess_garment,
        )

    def leffa_predict_pt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed):
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "pose_transfer",
            ref_acceleration,
            step,
            scale,
            seed,
        )

class LeffaService:
    def __init__(self):
        self.predictor = self._initialize_predictor()
        
    def _initialize_predictor(self):
        """初始化所有模型组件"""
        logger.info("Initializing models...")
        try:
            predictor = LeffaPredictor()
            logger.info("All models initialized successfully")
            return predictor
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError("Failed to initialize models")

leffa_service = LeffaService()

@app.post("/virtual-tryon",
         responses={
             200: {"content": {"image/png": {}}},
             400: {"description": "Invalid input"},
             500: {"description": "Internal server error"}
         })
async def virtual_tryon(
    person_image: Optional[UploadFile] = File(None, description="Person image file"),
    garment_image: Optional[UploadFile] = File(None, description="Garment image file"),
    person_image_base64: Optional[str] = Form(None, description="Person image as Base64 string"),
    garment_image_base64: Optional[str] = Form(None, description="Garment image as Base64 string"),
    model_type: Literal["viton_hd", "dress_code"] = Form("viton_hd", description="模型类型: viton_hd 或 dress_code"),
    garment_type: Literal["upper_body", "lower_body", "dresses"] = Form("upper_body", description="服装类型: upper_body, lower_body 或 dresses"),
    preprocess_garment: bool = Form(False, description="Enable garment preprocessing"),
    acceleration: bool = Form(False, description="Enable model acceleration"),
    steps: int = Form(30, ge=30, le=100),
    guidance_scale: float = Form(2.5, ge=0.1, le=5.0),
    seed: int = Form(42),
    repaint_mode: bool = Form(False)
) -> Response:
    """
    虚拟试衣接口 - 支持文件上传或Base64编码的图片
    """
    try:
        # 检查输入方式
        using_files = person_image is not None and garment_image is not None
        using_base64 = person_image_base64 is not None and garment_image_base64 is not None
        
        if not (using_files or using_base64):
            raise HTTPException(
                status_code=422, 
                detail="必须同时提供两张图片（通过文件上传或Base64字符串）"
            )
            
        # 处理图片并进行预测
        with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as person_tmp, \
             tempfile.NamedTemporaryFile(delete=True, suffix=".png") as garment_tmp:

            # 根据输入类型处理图片
            if using_files:
                # 文件上传方式 - 保持原有逻辑
                person_content = await person_image.read()
                person_tmp.write(person_content)
                person_tmp.flush()
                
                garment_content = await garment_image.read()
                garment_tmp.write(garment_content)
                garment_tmp.flush()
            else:
                # Base64方式 - 解码并保存到临时文件
                try:
                    # 处理可能的Data URL格式 (去除前缀如 "data:image/jpeg;base64,")
                    if "base64," in person_image_base64:
                        person_image_base64 = person_image_base64.split("base64,")[1]
                    if "base64," in garment_image_base64:
                        garment_image_base64 = garment_image_base64.split("base64,")[1]
                        
                    # 解码Base64字符串
                    person_content = base64.b64decode(person_image_base64)
                    person_tmp.write(person_content)
                    person_tmp.flush()
                    
                    garment_content = base64.b64decode(garment_image_base64)
                    garment_tmp.write(garment_content)
                    garment_tmp.flush()
                except Exception as e:
                    logger.error(f"Base64解码失败: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Base64解码失败: {str(e)}")

            # 其余的处理逻辑保持不变
            result = leffa_service.predictor.leffa_predict_vt(
                src_image_path=person_tmp.name,
                ref_image_path=garment_tmp.name,
                ref_acceleration=acceleration,
                step=steps,
                scale=guidance_scale,
                seed=seed,
                vt_model_type=model_type,
                vt_garment_type=garment_type,
                vt_repaint=repaint_mode,
                preprocess_garment=preprocess_garment
            )

            # 转换结果为字节流
            gen_image, mask, densepose = result
            
            # 将生成的图像转换为Base64
            image_bytes = numpy_to_bytes(gen_image)
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # 返回JSON响应而不是图像流，支持Web/微信小程序接收
            return {
                "image_base64": image_base64
            }
            
            # 如果需要直接返回图像，可以取消注释下面的代码
            # return Response(
            #     content=numpy_to_bytes(gen_image),
            #     media_type="image/png"
            # )

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pose-transfer",
         responses={
             200: {"content": {"image/png": {}}},
             400: {"description": "Invalid input"},
             500: {"description": "Internal server error"}
         })
async def pose_transfer(
    source_image: Optional[UploadFile] = File(None, description="源姿势图像文件"),
    reference_image: Optional[UploadFile] = File(None, description="参考人物图像文件"),
    source_image_base64: Optional[str] = Form(None, description="源姿势图像的Base64字符串"),
    reference_image_base64: Optional[str] = Form(None, description="参考人物图像的Base64字符串"),
    acceleration: bool = Form(False, description="启用模型加速"),
    steps: int = Form(30, ge=30, le=100, description="推理步数"),
    guidance_scale: float = Form(2.5, ge=0.1, le=5.0, description="引导比例"),
    seed: int = Form(42, description="随机种子")
) -> Response:
    """
    姿势迁移接口 - 支持文件上传或Base64编码的图片
    """
    try:
        # 检查输入方式
        using_files = source_image is not None and reference_image is not None
        using_base64 = source_image_base64 is not None and reference_image_base64 is not None
        
        if not (using_files or using_base64):
            raise HTTPException(
                status_code=422, 
                detail="必须同时提供两张图片（通过文件上传或Base64字符串）"
            )
            
        # 处理图片并进行预测
        with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as src_tmp, \
             tempfile.NamedTemporaryFile(delete=True, suffix=".png") as ref_tmp:

            # 根据输入类型处理图片
            if using_files:
                # 文件上传方式
                src_content = await source_image.read()
                src_tmp.write(src_content)
                src_tmp.flush()
                
                ref_content = await reference_image.read()
                ref_tmp.write(ref_content)
                ref_tmp.flush()
            else:
                # Base64方式 - 解码并保存到临时文件
                try:
                    # 处理可能的Data URL格式 (去除前缀如 "data:image/jpeg;base64,")
                    if "base64," in source_image_base64:
                        source_image_base64 = source_image_base64.split("base64,")[1]
                    if "base64," in reference_image_base64:
                        reference_image_base64 = reference_image_base64.split("base64,")[1]
                        
                    # 解码Base64字符串
                    src_content = base64.b64decode(source_image_base64)
                    src_tmp.write(src_content)
                    src_tmp.flush()
                    
                    ref_content = base64.b64decode(reference_image_base64)
                    ref_tmp.write(ref_content)
                    ref_tmp.flush()
                except Exception as e:
                    logger.error(f"Base64解码失败: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Base64解码失败: {str(e)}")

            # 执行姿势迁移预测
            result = leffa_service.predictor.leffa_predict_pt(
                src_image_path=src_tmp.name,
                ref_image_path=ref_tmp.name,
                ref_acceleration=acceleration,
                step=steps,
                scale=guidance_scale,
                seed=seed
            )

            # 转换结果为字节流
            gen_image, mask, densepose = result
            
            # 将生成的图像转换为Base64
            image_bytes = numpy_to_bytes(gen_image)
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # 返回JSON响应而不是图像流，支持Web/微信小程序接收
            return {
                "image_base64": image_base64
            }

    except ValueError as ve:
        logger.error(f"验证错误: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"姿势迁移失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples/list")
async def list_example_files():
    """获取示例文件列表"""
    try:
        return {
            "person1": list_dir("./ckpts/examples/person1"),
            "person2": list_dir("./ckpts/examples/person2"),
            "garment": list_dir("./ckpts/examples/garment")
        }
    except Exception as e:
        logger.error(f"Failed to list examples: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list examples")

@app.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)