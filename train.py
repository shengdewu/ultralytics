import os
import re
import random
import json
import subprocess
from ultralytics import YOLO
import argparse
import yaml


def probability(x):
    x = float(x)
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError("值必须在(0, 1)之间")
    return x


def parse_opt():
    parser = argparse.ArgumentParser(description='yolo11启动参数')
    parser.add_argument('--model-type', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型类型')
    parser.add_argument('--img-path', type=str, default='', help='图片路径')
    parser.add_argument('--label-file', type=str, default='', help='标注压缩文件名')
    parser.add_argument('--epochs', type=int, default=100, help='模型训练的epoch')
    parser.add_argument('--batch', type=int, default=16, help='模型输入的批数')
    parser.add_argument('--workers', type=int, default=4, help='数据处理的进程数')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='模型训练使用的设备')
    parser.add_argument('--imgsz', type=int, default=640, help='图片输入到模型的大小')
    parser.add_argument('--lr', type=probability, default=0.01, help='学习率取值范围(0, 1)')
    parser.add_argument("--save-dir", type=str, default="./save", help="结果保持路径")
    parser.add_argument("--resume", action="store_true", help="是否继续训练")
    parser.add_argument("--r-create", action="store_true", help="是否重新创建数据集合")
    return parser.parse_args()


def run_cmd(cmd: list):
    success = 0
    try:
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True)
        print(f"命令执行成功: {' '.join(cmd)}")
        if result.stdout:
            print(f"标准输出:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"支持命令{' '.join(cmd)}失败: {e}")
        success = 1
    return success


def create_data_cfg(opt):
    f = opt.label_file
    assert f.endswith('zip'), f'标签文件必须是zip文件，但是接受到的是{f}'
    assert os.path.exists(f), f'{f} 不存在'
    img_path = opt.img_path

    assert os.path.isabs(f) and os.path.isabs(img_path), '图片路径和标签路径必须是绝对路径'

    label_parent = f'{os.path.dirname(f)}/label_parent'
    os.makedirs(label_parent, exist_ok=True)

    label_path = f'{label_parent}/labels'
    if not os.path.exists(label_path) or opt.r_create:
        opt.r_create = True
        cmd = ['unzip', '-q', '-o', f, '-d', label_parent]
        status = run_cmd(cmd)
        assert status == 0, '解压失败'

    assert os.path.exists(label_path), '压缩文件里没有labels目录'
    note = f'{label_parent}/notes.json'
    assert os.path.exists(label_path), '压缩文件里没有发现 notes.json 文件'

    # 寻找公共目录作为数据的root目录
    label_arr = label_path.split('/')
    img_arr = img_path.split('/')
    cpt = -1
    for i in range(len(label_arr)):
        if label_arr[i] != img_arr[i]:
            break
        cpt = i

    assert cpt != -1, f'{label_path} 和 {img_path} 不在同级目录下'
    root_path = '/'.join(label_arr[:cpt+1])

    data_cfg = f'{root_path}/yolo.yaml'
    if os.path.exists(data_cfg) and not opt.r_create:
        return data_cfg

    label_names = [name.split('.') for name in os.listdir(label_path) if name.endswith('txt')]
    assert len(label_names) > 0, '压缩文件的labels目录没有发现标签文件[txt]'

    image_pattern = re.compile(
        r'.*\.(jpe?g|png|tiff)$',
        re.IGNORECASE  # 忽略大小写
    )
    img_names = [name.split('.') for name in os.listdir(img_path) if image_pattern.match(name)]

    assert len(img_names) > 0, f'图片目录{img_path}下没有发现图片(.*\\.(jpe?g|png|tiff)$)'

    label_rpath = label_path[label_path.find(root_path) + len(root_path) + 1:]
    img_rpath = img_path[img_path.find(root_path) + len(root_path) + 1:]

    label_names = dict(label_names)
    img_names = dict(img_names)

    data_sets = list()
    for name, l_suffix in label_names.items():
        i_suffix = img_names.get(name, None)
        if i_suffix is None:
            continue
        data_sets.append(f'{img_rpath}/{name}.{i_suffix},{label_rpath}/{name}.{l_suffix}')
    assert len(data_sets) > 0, f'{label_path} 和 {img_path} 没有相互匹配的图片和标签'

    random.shuffle(data_sets)

    sp_idx = int(0.85 * len(data_sets))
    assert sp_idx > 0, f'标注文件太少必须大于2张'
    with open(f'{root_path}/train.txt', mode='w') as f:
        for name in data_sets[:sp_idx]:
            f.write(f'{name}\n')

    with open(f'{root_path}/val.txt', mode='w') as f:
        for name in data_sets[sp_idx:]:
            f.write(f'{name}\n')

    yolo_cfg = dict()
    yolo_cfg['path'] = root_path
    yolo_cfg['train'] = 'train.txt'
    yolo_cfg['val'] = 'val.txt'
    yolo_cfg['names'] = list()

    with open(note, mode='r') as f:
        notes = json.load(f)['categories']
    for note in notes:
        yolo_cfg['names'].append({note['id']: note['name']})

    yaml_output = yaml.safe_dump(
        yolo_cfg,
        allow_unicode=True,  # 支持 Unicode
        default_flow_style=False,  # 使用块格式
        indent=2,
        sort_keys=False  # 保持原顺序
    )

    with open(data_cfg, 'w', encoding='utf-8') as f:
        f.write(yaml_output)

    return data_cfg


def run(opt):
    opt.data = create_data_cfg(opt)

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
