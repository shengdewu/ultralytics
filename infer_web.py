import os
from ultralytics import YOLO
from ultralytics.utils import ops
import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, confloat
import uvicorn
import base64
from contextlib import asynccontextmanager
import argparse
import gradio as gr
from gradio.themes.utils.colors import Color
from gradio.routes import mount_gradio_app


WEIGHT = 'save/weights/best.pt'
TASK = None


def base2numpy(base):
    """
    @param base:浏览器可识别的base64编码格式
    """
    img_bin = base64.b64decode(base.split(';base64,')[-1])
    img_buff = np.frombuffer(img_bin, dtype='uint8')
    image = cv2.imdecode(img_buff, -1)
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
        self.model = YOLO(weight, task=TASK)
        return

    def __call__(self, in_array: np.ndarray, conf=0.25):
        preds = self.model.simple_predict(in_array, conf=conf)

        results = list()
        preds = preds.detach().cpu().numpy()
        if preds.shape[1] == 7:
            # obb
            xyxyxyxy = ops.xywhr2xyxyxyxy(preds[:, :5])
            x = xyxyxyxy[..., 0]
            y = xyxyxyxy[..., 1]
            xyxy = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
            preds = np.concatenate([xyxy, preds[:, 5:]], axis=-1)
        for i in range(preds.shape[0]):
            pred = preds[i]
            x1, y1, x2, y2, conf, cls = pred
            results.append({
                "x1": x1.item(),
                "y1": y1.item(),
                "x2": x2.item(),
                "y2": y2.item(),
                "confidence": conf.item(),
                "cls": int(cls.item())
            })
        return results


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
    return app.state.engine_client(base2numpy(data.image_base64), data.confidence)


def infer_gradio(image, confidence):
    """
    图像处理测试函数
    """
    results = app.state.engine_client(image[:, :, ::-1], confidence)
    for result in results:
        cv2.rectangle(image, (int(result['x1']), int(result['y1'])), (int(result['x2']), int(result['y2'])),
                      color=(255, 255, 255))
        cls = result['cls']
        conf = result['confidence']
        cv2.putText(image, f'{cls}-{conf}', (int(result['x1']), int(result['y1'])), 1, 1, color=(255, 0, 0))
    return results, image


def cv_ui():
    zl_color = Color(name="zl_color", c50="#0055a2", c100="#0055a2", c200="#0055a2", c300="#0055a2", c400="#0055a2",
                     c500="#0055a2", c600="#0055a2", c700="#0055a2", c800="#0055a2", c900="#0055a2", c950="#0055a2", )

    with gr.Blocks(theme=gr.themes.Default(primary_hue=zl_color, secondary_hue="blue", spacing_size="sm", radius_size="sm")) as block:
        gr.Markdown("## 图像测试页面")
        gr.Markdown("改页面为图像测试页面，用户可以上传图像并获取处理结果。")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="上传图像", interactive=True, image_mode='RGB')
                confidence = gr.Slider(minimum=0, maximum=1.0, step=0.1, value=0.6, label='置信度阈值')
                submit_button = gr.Button("提交", variant='primary')
            with gr.Column():
                image_output = gr.Image(label='图片')
                result_output = gr.JSON(label="结果")

        submit_button.click(infer_gradio, inputs=(image_input, confidence), outputs=[result_output, image_output])

    return block


# 将 Gradio 应用挂载到 FastAPI 上，路径为 "/gradio"
app = mount_gradio_app(app=app, blocks=cv_ui(), path="/ui")


def parse_opt():
    parser = argparse.ArgumentParser(description='推理服务参数')
    parser.add_argument('--weight', type=str, default='/workspace/dataset/models/best.pt', help='模型路径')
    parser.add_argument('--task', type=str, default='', help='模型类型')
    return parser.parse_args()


# 启动命令
if __name__ == "__main__":
    args = parse_opt()
    WEIGHT = args.weight
    TASK = None if args.task == '' else args.task

    uvicorn.run(app, host="0.0.0.0", port=12345)
