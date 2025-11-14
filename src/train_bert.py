# src/train_bert.py
import argparse
import os

from src.data_utils import load_finchina_sa, time_based_split
from src.models.finbert_model import BertConfig, build_datasets, build_trainer


def main(args):
    df = load_finchina_sa(args.raw_dir)
    train_df, valid_df, test_df = time_based_split(df)

    config = BertConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        use_fp16=not args.no_fp16,
    )

    tokenizer, train_ds, valid_ds, test_ds = build_datasets(
        train_df, valid_df, test_df, config, config.model_name
    )

    trainer = build_trainer(train_ds, valid_ds, config)

    trainer.train()
    print("=== Evaluate on test set ===")
    metrics = trainer.evaluate(test_ds)
    print(metrics)

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw/finchina_sa")
    parser.add_argument("--model_name", type=str, default="yiyanghkust/finbert-tone-chinese")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=float, default=3)
    parser.add_argument("--output_dir", type=str, default="checkpoints/finbert")
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()
    main(args)
