import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast

from lightningnlp.callbacks import LoggingCallback
from lightningnlp.task.named_entity_recognition import (
    LEARNerDataModule,
    NamedEntityRecognitionTransformer,
)
from lightningnlp.task.utils import MODEL_MAP


# datasets args
pretrained_model_name_or_path = "hfl/chinese-roberta-wwm-ext"  # 预训练模型
train_batch_size = 16  # 训练集batch_size
validation_batch_size = 16  # 验证集batch_size
max_length = 256  # 序列最大长度
num_workers = 16  # 多进程加载数据

dataset_name = "datasets/cmeee"  # 训练数据所在目录
train_file = "train.json"  # 训练集文件名
validation_file = "dev.json"  # 验证集文件名
cache_dir = "datasets/cmeee"  # 数据缓存路径
task_name = "cmeee-bert-lear"  # 自定义任务名称

# model args
downstream_model_name = "lear"  # 模型名称
downstream_model_type = "bert"  # 预训练模型类型

# training args
learning_rate = 2e-5  # 学习率
base_model_name = MODEL_MAP[downstream_model_type][-1]  # 模型主干名称
other_learning_rate = 1e-4  # 除bert之外其他层的学习率
output_dir = "outputs/cmeee/lear"  # 模型保存路径

schema2prompt = {
    "dis": "疾病，主要包括疾病、中毒或受伤和器官或细胞受损",
    "sym": "临床表现，主要包括症状和体征",
    "pro": "医疗程序，主要包括检查程序、治疗或预防程序",
    "equ": "医疗设备，主要包括检查设备和治疗设备",
    "dru": "药物，是用以预防、治疗及诊断疾病的物质",
    "ite": "医学检验项目，是取自人体的材料进行血液学、细胞学等方面的检验",
    "bod": "身体，主要包括身体物质和身体部位",
    "dep": "部门科室，医院的各职能科室",
    "mic": "微生物类，一般是指细菌、病毒、真菌、支原体、衣原体、螺旋体等八类微生物",
}


def main():
    pl.seed_everything(seed=42)

    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)

    dm = LEARNerDataModule(
        schema2prompt=schema2prompt,
        tokenizer=tokenizer,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        num_workers=num_workers,
        dataset_name=dataset_name,
        train_file=train_file,
        validation_file=validation_file,
        max_length=max_length,
        cache_dir=cache_dir,
        task_name=task_name,
        is_chinese=True,
    )

    model = NamedEntityRecognitionTransformer(
        downstream_model_name=downstream_model_name,
        downstream_model_type=downstream_model_type,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        tokenizer=tokenizer,
        labels=dm.label_list,
        average="micro",
        learning_rate=learning_rate,
        base_model_name=base_model_name,
        other_learning_rate=other_learning_rate,
        output_dir=output_dir,
    )

    model_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="best_model",
        monitor="val_f1_micro",
        save_top_k=1,
        mode="max",
    )

    wandb_logger = WandbLogger(project="Named Entity Recognition", name=task_name)

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
