from ultralytics import YOLO
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoint/yolo11s.pt", help="initial weights path")
    parser.add_argument("--data", type=str, default="open-images-v7.yaml")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--save_dir", type=str, default="", help="result save path")
    parser.add_argument("--resume", action="store_true", help="resume")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--workers", type=int, default=8, help="workers")
    parser.add_argument("--epochs", type=int, default=200, help="epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="imgsz")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="(float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)")
    parser.add_argument("--lrf", type=float, default=0.01, help="(float) final learning rate (lr0 * lrf)")
    return parser.parse_args()


def run(opt):
    # 加载模型
    model = YOLO(opt.weights)

    # 训练模型
    train_results = model.train(
        data=opt.data,  # 数据集 YAML 路径
        epochs=opt.epochs,  # 训练轮次
        imgsz=opt.imgsz,  # 训练图像尺寸
        device=opt.device,  # 运行设备，例如 device=0 或 device=0,1,2,3 或 device=cpu
        batch=opt.batch,
        resume=opt.resume,
        save_dir=opt.save_dir,
        workers=opt.workers,
        lr0=opt.lr0,
        lrf=opt.lrf
    )

    # 评估模型在验证集上的性能
    metrics = model.val()

    # 在图像上执行对象检测
    # results = model("/home/ros-ms/workspace/datasets/coco128-seg/images/train2017/000000000138.jpg")
    # results[0].show()

    # 将模型导出为 ONNX 格式
    # path = model.export(format="torchscript")  # 返回导出模型的路径
    return


if __name__ == '__main__':
    run(parse_opt())
