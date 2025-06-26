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


def client(img_path, address='http://127.0.0.1:8000/infer'):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    buffer = numpy2base(image)

    params = {
        "image_base64": buffer,
        "confidence": 0.6
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
            cv2.rectangle(image, (int(result['x1']), int(result['y1'])), (int(result['x2']), int(result['y2'])), color=(255, 255, 255))
            cls = result['cls']
            conf = result['confidence']
            cv2.putText(image, f'{cls}-{conf}', (int(result['x1']), int(result['y1'])), 1, 1, color=(255, 0, 0))
        cv2.imwrite('result.jpg', image)

    return


# 启动命令
if __name__ == "__main__":
    client('./datasets/0-2.jpg', 'http://10.113.19.30:38000/infer')
