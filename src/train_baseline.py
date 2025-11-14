import os
import argparse
import joblib

from src.data_utils import load_finchina_sa, time_base_split
from src.models.baseline import BaselineConfig, build_tfidf_lr, evaluate_baseline

def main(args):
    df = load_finchina_sa(args.raw_dir)
    train_df, valid_df, test_df = time_base_split(df)
    
    config = BaselineConfig()
    model = build_tfidf_lr(config)
    
    print("Train size:", len(train_df), " Valid size:", len(valid_df), " Test size", len(test_df))
    
    model.fit(train_df["text"], train_df["label"])
    print("=== Validation performance ===")
    evaluate_baseline(model, valid_df["text"], valid_df["label"])
    
    print("=== Test performance ===")
    evaluate_baseline(model, test_df["text"], test_df["label"])
    
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.output_dir, "tfidf_lr.joblib"))
    print("Model saved to", os.path.join(args.output_dir, "tfidf_lr.joblib"))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw/finchina_sa")
    parser.add_argument("--output_dir", type=str, default="checkpoints/baseline")
    args = parser.parse_args()
    main(args)