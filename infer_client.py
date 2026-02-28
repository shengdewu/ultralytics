import os

import cv2
import json
import requests
import base64


def numpy2base(image):
    """
    @param image:numpy数组格式图片
    """
    success, encoded_image = cv2.imencode(".jpg", image)
    byte_data = encoded_image.tobytes()
    base = "data:image;base64," + base64.b64encode(byte_data).decode('utf-8')
    return base


def client(img_file, out_file, address='http://127.0.0.1:8000/infer'):
    image = cv2.imread(img_file, cv2.IMREAD_COLOR)
    buffer = numpy2base(image)

    params = {
        "image_base64": buffer,
        "confidence": 0.1
    }

    results = None
    try:
        response = requests.post(address, json=params, timeout=10)
        if response.status_code != 200:
            print(f'获取推理结果 出错{response.text}')
        else:
            results = response.json()
            print(results)
    except Exception as err:
        print(f'获取推理结果 出错:{err}')

    if results is not None:
        for result in results:
            cls = result['cls']
            conf = result['confidence']
            color = (0, 255, 0)
            if conf >= 0.65:
                color = (0, 0, 255)
            cv2.putText(image, f'{cls}-{conf:.2}', (int(result['x1']), int(result['y1'])), 2, 4, color=color, thickness=2)
            cv2.rectangle(image, (int(result['x1']), int(result['y1'])), (int(result['x2']), int(result['y2'])),
                          color=color, thickness=3)
        cv2.imwrite(out_file, image)

    return


# 启动命令
if __name__ == "__main__":
    client('./datasets/0-2.jpg', './datasets/0-2-result.jpg', 'http://127.0.0.1:12345/infer')
