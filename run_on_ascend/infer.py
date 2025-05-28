import os
import math
import numpy as np
import cv2
import time
from ascend_model import OMNet
import torch
import tqdm
import sys

import ops


class Yolo:
    def __init__(self, mode_path='', device=0):
        self.net = OMNet(mode_path, device)
        return

    @staticmethod
    def pad32(img):
        height, width = img.shape[:2]
        pad_h = math.ceil(height / 32) * 32
        pad_w = math.ceil(width / 32) * 32

        if (pad_h, pad_w) != (height, width):
            h_offset = pad_h - height
            w_offset = pad_w - width
            h_pad_top = h_offset // 2
            h_pad_bottom = h_offset - h_pad_top
            w_pad_left = w_offset // 2
            w_pad_right = w_offset - w_pad_left

            img = cv2.copyMakeBorder(img, top=h_pad_top, bottom=h_pad_bottom,
                                     left=w_pad_left, right=w_pad_right,
                                     borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img

    @staticmethod
    def run_onnx(in_img):
        import onnxruntime

        model = onnxruntime.InferenceSession('yolo11l-fp16.onnx',
                                             providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        param = dict()
        param[model.get_inputs()[0].name] = in_img

        result = model.run(None, param)
        return result

    def run(self, in_path, out_path):
        end_time = 0
        data_time = 0
        for name in tqdm.tqdm(os.listdir(in_path)):
            if not name.endswith('jpg'):
                continue
            dt, et = self.__call__(f'{in_path}/{name}', f'{out_path}/{name}')
            data_time += dt
            end_time += et
        print(f'cost data {data_time}, mode {end_time}')
        return

    def __call__(self, img_path, out_path):
        start = time.time()
        ori_im = cv2.imread(img_path)
        im = self.pad32(ori_im)
        im = cv2.resize(im, (640, 640))
        im = np.stack([im])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im.astype(np.float32))
        im /= 255
        data_time = time.time()-start
        # dt = self.run_onnx(im)[0]

        results = self.net.forward(im)
        if len(results) == 0:
            print('输出为空')
            return

        end = time.time() - start

        start = time.time()
        dt = np.array(results, dtype=np.float32)
        dt = dt.reshape(1, 80 + 4, -1)
        # print(dt.shape)
        # np.save('pred.npy', dt)
        preds = ops.non_max_suppression(torch.from_numpy(dt),
                                        0.25,
                                        0.5,
                                        agnostic=False,
                                        max_det=300,
                                        classes=None)
        no_model_time = time.time() - start + data_time
        # print(preds)
        ori_ims = [ori_im]
        for pred, ori_im in zip(preds, ori_ims):
            pred[:, :4] = ops.scale_boxes(im.shape[2:], pred[:, :4], ori_im.shape)
            pred = pred.numpy()
            for i in range(pred.shape[0]):
                x1, y1, x2, y2 = pred[i, :4].astype(np.int32)
                s, c = pred[i, 4:]
                cv2.rectangle(ori_im, (x1, y1), (x2, y2), color=(int(c), 0, int(c * 255 / 80)))
                cv2.putText(ori_im, f'{s}-{c}', (x1, y1), 1, 1, color=(int(c), 0, int(c * 255 / 80)))

            cv2.imwrite(out_path, ori_im)
            break
        return no_model_time, end


if __name__ == '__main__':
    om = sys.argv[1]

    yolo = Yolo(om, 0)
    yolo('test.jpg', 'om.jpg')

    # in_path = '/data/models/yolo11/dq/belt0217a/'
    # out_path = '/data/models/yolo11/results'
    # yolo.run(in_path, out_path)
