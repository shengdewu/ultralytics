from ultralytics import YOLO
import torch
import time
import torch
from thop import profile
import time
import os

import argparse


def test_performance(model: torch.nn.Module, in_size):
    model.eval()

    flops, params = profile(model, inputs=(torch.ones((1, 3, in_size, in_size), dtype=torch.float32),), verbose=False)
    total_time = 0
    cnt = 10
    for i in range(cnt):
        in_tensor = torch.randn((1, 3, in_size, in_size), dtype=torch.float32)
        start_time = time.time()
        model(in_tensor)
        total_time += time.time() - start_time

    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    print(f'{model.__class__.__name__} FLOPS = {flops / 1000 ** 3}G, Params = {params / 1000 ** 2}M, Size = {(param_size+ buffer_size)/1024/1024}M, TIME {total_time/cnt}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yolo11启动参数')
    parser.add_argument('--model', type=str, default='checkpoints/yolo11l.pth', help='模型类型')
    args = parser.parse_args()

    model = YOLO(args.model)  # load an official model

    # results = model.predict("read.jpg", conf=0.25)
    # results[0].show()
    
    # Export the model
    model.export(format="onnx", opset=11)

    test_performance(model, 640)
