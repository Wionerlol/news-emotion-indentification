news-sentiment-cn/
│
├─ README.md
├─ requirements.txt
├─ config.py
│
├─ data/
│   ├─ raw/
│   │   └─ finchina_sa/          # 解压后的原始数据
│   └─ processed/
│       └─ finchina_sa.csv       # 清洗好的三分类数据（正/负/中性）
│
├─ src/
│   ├─ __init__.py
│   │
│   ├─ data_utils.py             # 读原始数据、清洗、时间切分、类不平衡统计
│   │
│   ├─ models/
│   │   ├─ __init__.py
│   │   ├─ baseline.py           # TF-IDF + LR/SVM 基线
│   │   └─ finbert_model.py      # FinBERT / BERT 中文模型封装
│   │
│   ├─ train_baseline.py         # 训练 & 评估传统 ML 模型
│   ├─ train_bert.py             # 训练 & 评估 BERT/FinBERT
│   └─ eval_ic.py                # 可选：情绪得分与未来收益 IC
│
└─ notebooks/
    ├─ 01_prepare_data.ipynb     # Colab：下载数据 + 预处理
    └─ 02_train_bert_colab.ipynb # Colab：GPU 上训练 BERT/FinBERT
