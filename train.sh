#!/bin/bash

docker run --rm --gpus='"device=0"' --cpus=8 --name ultralytics --shm-size=20g \
-v /mnt/sda/:/mnt/sda/ -v /home/thinkbook/workspace:/home/thinkbook/workspace \
-v /mnt/sdb/train.output:/mnt/sdb/train.output \
-t aihub.chinalco.com.cn:30102/cdsw/ultralytics:torch2.11-cuda12.8 \
--model-type 11s-obb \
--data /mnt/sda/datasets/皮带撕裂数据集合-目标检测/yc.yaml \
--save-dir /mnt/sdb/train.output/ultralytics/皮带异常-11s-obb-test \
--pretrain-model /home/thinkbook/workspace/ultralytics/checkpoints/yolo11s-obb.pt