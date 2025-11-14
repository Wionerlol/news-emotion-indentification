from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# TODO: dataclass是什么，pipeline是怎么用的，TfidfVectorizer里面的参数是什么意思，LogisticRegression里面的参数是什么意思
@dataclass
class BaselineConfig:
    max_features: int = 50000
    ngram_range: Tuple[int, int] = (1, 2)
    c: float = 4.0
    class_weight: str = "balanced"
    

def build_tfidf_lr(config: BaselineConfig) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features = config.max_features,
        ngram_range = config.ngram_range
    )
    
    clf = LogisticRegression(
        C = config.c,
        max_iter = 200,
        class_weight = config.class_weight,
        n_jobs = -1
    )
    
    model = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])
    
    return model


def evaluate_baseline(model: Pipeline, X, y_true):
    y_pred = model.predict(X)
    print(classification_report(y_true, y_pred, digits=4))