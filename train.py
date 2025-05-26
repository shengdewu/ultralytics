from ultralytics import YOLO
import argparse


def probability(x):
    x = float(x)
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError("值必须在(0, 1)之间")
    return x


def parse_opt():
    parser = argparse.ArgumentParser(description='yolo11启动参数')
    parser.add_argument('--model-type', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型类型')
    parser.add_argument('--data', type=str, default='', help='数据配置文件 yaml 格式')
    parser.add_argument('--epochs', type=int, default=100, help='模型训练的epoch')
    parser.add_argument('--batch', type=int, default=16, help='模型输入的批数')
    parser.add_argument('--workers', type=int, default=4, help='数据处理的进程数')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='模型训练使用的设备')
    parser.add_argument('--imgsz', type=int, default=640, help='图片输入到模型的大小')
    parser.add_argument('--lr', type=probability, default=0.01, help='学习率取值范围(0, 1)')    
    parser.add_argument("--save-dir", type=str, default="./save", help="结果保持路径")
    parser.add_argument("--resume", action="store_true", help="是否继续训练")
    return parser.parse_args()


def run(opt):

    opt.weights = f'ultralytics/yolo11-ckpt/yolo11{opt.model_type}.pt'

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
        lr0=opt.lr,
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
