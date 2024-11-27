#!/bin/bash

# 激活 conda 环境
conda activate cmx

# 获取当前时间并格式化为文件名的一部分
timestamp=$(date +%Y%m%d_%H%M%S)

# 使用时间戳生成日志文件名
log_file="log_${timestamp}.txt"

# 检查是否提供了损失函数参数
if [ -z "$1" ]; then
  echo "Usage: $0 <loss_function>"
  echo "Available loss functions: FocalBCEDiceLoss, BCEDiceLoss, BCETverskyLoss, DiceLoss"
  exit 1
fi

# 获取损失函数参数
loss_function=$1

# 运行训练脚本并将输出重定向到日志文件
nohup python3 new_train.py --loss_function $loss_function > $log_file 2>&1 &
