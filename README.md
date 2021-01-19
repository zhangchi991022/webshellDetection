# webshellDetection
基于机器学习的webshell检测工具
## 简介：

提取PHP执行中的opcode，采用 opcode调用序列 + tf-词汇表模型进行关键信息提取

采用CNN算法和MLP算法进行训练，进行 PHP WebShell 的检测。

## PHP opcode调用：

```
1. 下载 vld.dll 插件并存放在php ext 目录下
2. 配置 php.ini 激活vld.dll 文件
```

## 训练：

将正常webshell放入white-list中，将webshell放入black-list中。

python train.py

## 对比：
python showDifferences.py 

可视化对比MLP算法和CNN算法的识别性能


## 检测：

检测单个文件：

python check.py
