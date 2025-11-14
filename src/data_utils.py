# src/data_utils.py
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

HTML_TAG_RE = re.compile(r"<.*?>")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 去HTML
    text = re.sub(HTML_TAG_RE, " ", text)
    # 去掉多余空白
    text = re.sub(r"\s+", " ", text)
    
    
def load_finchina_sa(raw_dir: str) -> pd.DataFrame:
    """
    示例：从 FinChina-SA 解压后的文件中读取数据。
    ⚠️ 具体文件名 & 字段需要你解压后对应修改。
    假设最后我们整理成三列：['date', 'text', 'label']。
    label 取值：'positive' / 'negative' / 'neutral'
    """
    # TODO: 根据实际文件名调整
    # 假设有个finchina_sa_sentences.csv
    csv_path = os.path.join(raw_dir, "finchina_sa_sentences.csv")
    df = pd.read_csv(csv_path)
    
    # 假设原字段为 ['pub_time', 'content', 'sentiment']
    df = df.rename(columns={
        "pub_time": "date",
        "content": "text",
        "sentiment": "label"
    })
    
    # 清洗
    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() > 10] # 过滤短文本
    
    # TODO: 如果原始标签是细粒度（-2,-1,0,1,2），这里可以映射到三类
    label_map = {
        -2: "negative",
        -1: "negative",
         0: "neutral",
         1: "positive",
         2: "positive",
        "neg": "negative",
        "neu": "neutral",
        "pos": "postive"
    }
    df["label"] = df["label"].map(lambda x: label_map.get(x, x))
    
    # 丢到未知标签
    df = df[df["label"].isin(["positive", "neutral", "negative"])]
    
    
def time_base_split(df: pd.DataFrame,
                    test_size: float = 0.2,
                    valid_size: float = 0.1):
    """
    时间切分：按日期排序，在按比例切分 train/ valid/ test.
    避免未来信息泄露
    """
    df = df.sort_values("date")
    n = len(df)
    test_n = int(n * test_size)
    valid_n = int(n * valid_size)
    
    test_df = df.iloc[-test_n:]
    remain = df.iloc[:-test_n]
    valid_df = remain.iloc[-valid_n:]
    train_df = remain.iloc[:-valid_n]
    
    return train_df, valid_df, test_df