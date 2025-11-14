from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class BertConfig:
    model_name: str = "yiyanghkust/finbert-tone-chinese"
    max_length: int = 256
    lr: float = 2e-5
    batch_size: int = 16
    num_epoch: int = 3
    weight_decay: float = 0.01
    output_dir: str = "checkpoints/finbert"
    use_fp16: bool = True
    

LABEL2ID = {"neutral": 0, "positive": 1, "ngetive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def build_dataset(train_df, valid_df, test_df, config: BertConfig, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def encode_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length
        )
    
    def df_to_dataset(df):
        return Dataset.from_pandas(df[["text", "label"]])
    
    def add_labels(example):
        example["labels"] = LABEL2ID[example["label"]]
        return example
    
    train_ds = df_to_dataset(train_df).map(add_labels)
    valid_ds = df_to_dataset(valid_df).map(add_labels)
    test_ds = df_to_dataset(test_df).map(add_labels)
    
    train_ds = train_ds.map(encode_batch, batched=True)
    valid_ds = valid_ds.map(encode_batch, batched=True)
    test_ds = test_ds.map(encode_batch, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    valid_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    return tokenizer, train_ds, valid_ds, test_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    # AUC 只能在二分类或 one-vs-rest 设置中用，这里先只返回 F1
    return {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
    

def build_trainer(train_ds, valid_ds, config: BertConfig, num_labels: int = 3):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        fp16=config.use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    return trainer