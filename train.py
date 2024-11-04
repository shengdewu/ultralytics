from ultralytics import YOLO

if __name__ == '__main__':

    #加载模型
    model = YOLO("checkpoint/yolo11s-seg.pt")

    # 训练模型
    train_results = model.train(
        data="rpc.yaml",  # 数据集 YAML 路径
        epochs=200,  # 训练轮次
        imgsz=640,  # 训练图像尺寸
        device="0",  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
        batch=32,
        resume=False,
        save_dir='/home/ros-ms/train.output/segment/yolo11'
    )

    # 评估模型在验证集上的性能
    metrics = model.val()

    # 在图像上执行对象检测
    # results = model("/home/ros-ms/workspace/datasets/coco128-seg/images/train2017/000000000138.jpg")
    # results[0].show()

    # 将模型导出为 ONNX 格式
    # path = model.export(format="torchscript")  # 返回导出模型的路径