import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast

from lightningnlp.callbacks import LoggingCallback
from lightningnlp.task.text_classification import (
    TextClassificationDataModule,
    TextClassificationTransformer,
)


# datasets hyperparameters
pretrained_model_name_or_path = "hfl/chinese-roberta-wwm-ext"  # 预训练模型
train_batch_size = 16  # 训练集batch_size
validation_batch_size = 32  # 验证集batch_size
max_length = 256  # 序列最大长度
num_workers = 4  # 多进程加载数据

# 方式1：从huggingface下载数据集
dataset_name = "xusenlin/tnews"

# # 方式2：使用自定义数据集
# dataset_name = "datasets/cmeee"  # 训练数据所在目录
# train_file = "train.json"  # 训练集文件名
# validation_file = "dev.json"  # 验证集文件名

cache_dir = "datasets/tnews"  # 数据缓存路径
task_name = "tnews-bert-fc"  # 自定义任务名称

# model hyperparameters
downstream_model_name = "fc"  # 模型名称
downstream_model_type = "bert"  # 预训练模型类型

# training hyperparameters
learning_rate = 2e-5  # 学习率
output_dir = "outputs/tnews/fc"  # 模型保存路径


def main():
    pl.seed_everything(seed=42)

    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)

    dm = TextClassificationDataModule(
        tokenizer=tokenizer,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        dataset_name=dataset_name,
        num_workers=num_workers,
        # train_file=train_file,  # 自定义数据集最好指定训练集和验证集文件名
        # validation_file=validation_file,
        max_length=max_length,
        cache_dir=cache_dir,
    )

    model = TextClassificationTransformer(
        downstream_model_name=downstream_model_name,
        downstream_model_type=downstream_model_type,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        tokenizer=tokenizer,
        labels=dm.label_map,
        learning_rate=learning_rate,
        output_dir=output_dir,
    )

    model_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="best_model",
        monitor="val_accuracy",
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(project="Text Classification", name=task_name)

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        max_epochs=12,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        callbacks=[model_ckpt, LoggingCallback()]
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
