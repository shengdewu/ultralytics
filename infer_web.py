import os
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, confloat
import uvicorn
import base64
from contextlib import asynccontextmanager
import argparse

WEIGHT = 'save/weights/best.pt'


def base2numpy(base):
    """
    @param base:浏览器可识别的base64编码格式
    """
    img_bin = base64.b64decode(base.split(';base64,')[-1])
    img_buff = np.frombuffer(img_bin, dtype='uint8')
    image = cv2.imdecode(img_buff, 1)
    return image


def numpy2base(image):
    """
    @param image:numpy数组格式图片
    """
    success, encoded_image = cv2.imencode(".jpg", image)
    byte_data = encoded_image.tobytes()
    base = "data:image;base64," + base64.b64encode(byte_data).decode('utf-8')
    return base


class YOLOEngine:
    def __init__(self, weight: str):
        self.model = YOLO(weight)
        return

    def __call__(self, in_array: np.ndarray, conf=0.25):
        return self.model.simple_predict(in_array, conf=conf)


# 启动时初始化模型
@asynccontextmanager
async def lifespan(app: FastAPI):
    assert os.path.exists(WEIGHT), f'{WEIGHT}不存在'
    # 启动时初始化资源
    app.state.engine_client = YOLOEngine(WEIGHT)

    yield  # 应用运行期间


app = FastAPI(title="推理服务", lifespan=lifespan)


class InfData(BaseModel):
    image_base64: str
    confidence: confloat(ge=0.0, le=1.0) = 0.6


# 处理文本+图像输入的 API
@app.post("/infer")
async def infer(data: InfData):
    results = app.state.engine_client(base2numpy(data.image_base64), data.confidence)

    results = results.detach().cpu().numpy()[0]
    x1, y1, x2, y2, conf, cls = results

    # 返回结果
    return {
        "x1": x1.item(),
        "y1": y1.item(),
        "x2": x2.item(),
        "y2": y2.item(),
        "confidence": conf.item(),
        "cls": int(cls.item())
    }


def parse_opt():
    parser = argparse.ArgumentParser(description='推理服务参数')
    parser.add_argument('--weight', type=str, default='', help='模型路径')
    return parser.parse_args()


# 启动命令
if __name__ == "__main__":
    args = parse_opt()
    WEIGHT = args.weight

    uvicorn.run(app, host="0.0.0.0", port=8000)
