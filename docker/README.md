# lightningnlp的Docker安装

## 构建镜像

```
sudo docker build -t lightningnlp-gpu:cuda11.3 .
```

## 运行

```
sudo docker run -it --name="lightningnlp-gpu" --gpus=all lightningnlp-gpu:cuda11.3
```

## 测试

进入容器后运行 `test` 文件中的示例测试是否安装成功