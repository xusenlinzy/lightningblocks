import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast

from lightningnlp.callbacks import LoggingCallback
from lightningnlp.task.uie import (
    UIEDataModule,
    UIEModel,
)

# datasets args
pretrained_model_name_or_path = "uie_base_pytorch"  # 预训练模型
train_batch_size = 8  # 训练集batch_size
validation_batch_size = 8  # 验证集batch_size
train_max_length = 512  # 序列最大长度
num_workers = 16  # 多进程加载数据

dataset_name = "datasets/medical"
train_file = "train.json"  # 训练集文件名
validation_file = "dev.json"  # 验证集文件名
cache_dir = "datasets/medical"  # 数据缓存路径
task_name = "uie"  # 自定义任务名称

# model args
downstream_model_name = "uie"  # 模型名称
downstream_model_type = "ernie"  # 预训练模型类型

# training args
learning_rate = 1e-5  # 学习率
output_dir = "outputs/uie/uie-medical"  # 模型保存路径


def main():
    pl.seed_everything(seed=42)

    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)

    dm = UIEDataModule(
        tokenizer=tokenizer,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        num_workers=num_workers,
        dataset_name=dataset_name,
        train_file=train_file,
        validation_file=validation_file,
        cache_dir=cache_dir,
        train_max_length=train_max_length,
        task_name=task_name,
    )

    model = UIEModel(
        downstream_model_name=downstream_model_name,
        downstream_model_type=downstream_model_type,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        tokenizer=tokenizer,
        learning_rate=learning_rate,
        output_dir=output_dir,
    )

    model_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="best-model",
        monitor="val_f1",
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(project="UIE", name=task_name)

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        max_epochs=10,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        callbacks=[model_ckpt, LoggingCallback()]
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
