import os
import re
import random
import json
import subprocess
import argparse
import yaml
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load, guess_model_name, guess_model_type, guess_model_scale_by_type


MODEL_CFG_PATH = 'ultralytics/cfg/models'


def probability(x):
    x = float(x)
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError("值必须在(0, 1)之间")
    return x


def collect_mode_type():
    mtype = ['11', 'v3', 'v5', 'v6', 'v8', 'v9', 'v10']
    model_type = list()
    for m in mtype:
        for yml in os.listdir(f'{MODEL_CFG_PATH}/{m}'):
            version, obb = guess_model_type(yml)
            if version != '':
                model_type.append(f'{m}{version}{obb}')
            else:
                model_type.append(m)
    return model_type


def device_type(value):
    # 正则匹配 cuda:<数字> 或 cpu
    if value == 'cpu' or value == 'cuda' or re.fullmatch(r'cuda:\d+', value):
        return value
    else:
        raise argparse.ArgumentTypeError(f"Invalid device: {value}")


def parse_opt():
    parser = argparse.ArgumentParser(description='yolo目标检测启动参数')
    parser.add_argument('--model-type', type=str, default='11s-obb',
                        choices=collect_mode_type(),
                        help='模型类型，目前支持11, v3, v5, v6, v8, v9, v10的各种版本')
    parser.add_argument('--pretrain-model', type=str, default='',
                        help='预训练模型必须和model-type匹配 /xx/yolo11n.pt，可以从这里下载对应版本https://docs.ultralytics.com/models/'
                             'resume=True时，这个参数会失效')
    parser.add_argument('--data', type=str, default='', help='yaml格式数据配置, 如果使用了此配置,则--img-path和--label-file失效')
    parser.add_argument('--img-path', type=str, default='', help='图片路径')
    parser.add_argument('--label-file', type=str, default='', help='标签目录(必须包含labels和notes.json）或者压缩文件 (就是在标签目录下的zip)')
    parser.add_argument('--epochs', type=int, default=100, help='模型训练的epoch')
    parser.add_argument('--batch', type=int, default=8, help='模型输入的批数')
    parser.add_argument('--workers', type=int, default=4, help='数据处理的进程数')
    parser.add_argument('--device', type=device_type, default='cuda', help='模型训练使用的设备')
    parser.add_argument('--imgsz', type=int, default=640, help='图片输入到模型的大小')
    parser.add_argument('--lr', type=probability, default=0.005, help='学习率取值范围(0, 1)')
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
        print(f"执行命令{' '.join(cmd)} 失败: {e}")
        success = 1
    return success


def create_data_cfg(opt):
    if opt.data != '':
        return opt.data

    f = opt.label_file
    assert os.path.exists(f), f'{f} 不存在'
    img_path = opt.img_path

    assert os.path.isabs(f) and os.path.isabs(img_path), '图片路径和标签路径必须是绝对路径'

    if f.endswith('zip'):
        label_parent = f'{os.path.dirname(f)}/label_parent'
        os.makedirs(label_parent, exist_ok=True)

        label_path = f'{label_parent}/labels'
        if not os.path.exists(label_path) or opt.r_create:
            if os.path.exists(label_path):
                run_cmd(['rm', '-rf', label_path])
            opt.r_create = True
            cmd = ['unzip', '-q', '-o', f, '-d', label_parent]
            status = run_cmd(cmd)
            assert status == 0, '解压失败'

        note = f'{label_parent}/notes.json'
    else:
        label_path = f'{f}/labels'
        note = f'{f}/notes.json'

    assert os.path.exists(label_path), '压缩文件里没有labels目录'
    assert os.path.exists(note), '压缩文件里没有发现 notes.json 文件'

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

    scale = guess_model_scale_by_type(opt.model_type)
    model_version = opt.model_type[:opt.model_type.find(scale)] if scale != '' else opt.model_type

    model_name = f'yolo{opt.model_type}'
    opt.weights = f'{MODEL_CFG_PATH}/{model_version}/{model_name}.yaml'

    print(f'加载模型配置文件 {opt.weights } !')

    if not opt.resume and os.path.exists(opt.pretrain_model):
        check_model_name = guess_model_name(opt.pretrain_model)
        if check_model_name == '':
            check_model_name = opt.pretrain_model.split('/')[-1].split('.')[0]
            assert check_model_name in ['yolov3', 'yolov6']
        assert check_model_name == model_name, f'({opt.pretrain_model})预训练模型和指定的模型类型不匹配({model_name})'

        opt.weights = opt.pretrain_model
        print(f'使用预训练模型 {opt.pretrain_model} !')

    if opt.resume:
        resume_weight = f'{opt.save_dir}/weights/last.pt' # 和 ultralytics/engine/trainer.py:117 强关联
        assert os.path.exists(resume_weight), f'启动恢复训练时,最近一次保存的checkpoint({resume_weight})不存在'
        check = torch.load(resume_weight, map_location='cpu')
        check_model = check.get('model', None)
        check_model = check_model if check_model is not None else check.get('ema', None)
        assert check_model is not None, f'{resume_weight} 模型格式不正确，这个模型不能进行resume, 请重新训练'
        if opt.weights.endswith('yaml'):
            weight = yaml_model_load(opt.weights)
        else:
            weight = torch.load(opt.weights, map_location='cpu')['model'].yaml
        check = check_model.yaml
        for k, cv in check.items():
            if k in ['yaml_file', 'ch', 'nc']:
                continue
            wv = weight.get(k, None)
            assert wv is not None and wv == cv, f'对比恢复训练的模型和[model_type]指定的模型类型({opt.model_type})时，发现{k}的' \
                                                f'值不一致({cv})!=({wv})不一致, 请检查输入参数(model_type)和原来的是否一致!'

        opt.weights = resume_weight

    # 加载模型
    model = YOLO(opt.weights, task='detect')

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
