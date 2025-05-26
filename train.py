import argparse

from ultralytics import YOLO


def probability(x):
    x = float(x)
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError("值必须在(0, 1)之间")
    return x


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='yolo11启动参数')
    parser.add_argument('--model-type', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型类型')
    parser.add_argument('--data', type=str, default='', help='数据配置文件 yaml 格式')
    parser.add_argument('--epochs', type=int, default=100, help='模型训练的epoch')
    parser.add_argument('--batch', type=int, default=16, help='模型输入的批数')
    parser.add_argument('--workers', type=int, default=4, help='数据处理的进程数')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='模型训练使用的设备')
    parser.add_argument('--imgsz', type=int, default=640, help='图片输入到模型的大小')
    parser.add_argument('--lr', type=probability, default=0.01, help='学习率取值范围(0, 1)')

    args = parser.parse_args()

    checkpoints = f'ultralytics/yolo11-ckpt/yolo11{args.model_type}.pt'

    print(f'接受到的参数 {args}')
    model = YOLO(checkpoints)

    results = model.train(data=args.data,
                          epochs=args.epochs,
                          batch=args.batch,
                          workers=args.workers,
                          device=args.device,
                          lr0=args.lr,
                          imgsz=args.imgsz)