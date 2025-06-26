#! /bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "启动服务..."

python3 train.py $@