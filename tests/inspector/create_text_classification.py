import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
twenty_train = fetch_20newsgroups(
    subset="train",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers"),
)
twenty_test = fetch_20newsgroups(
    subset="test",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers"),
)
df_train = pd.DataFrame(
    data=np.c_[twenty_train.data, twenty_train.target], columns=["data", "target"]
)
df_test = pd.DataFrame(data=np.c_[twenty_test.data, twenty_test.target], columns=["data", "target"])

text_vectorizer = TfidfVectorizer(min_df=3, stop_words="english", ngram_range=(1, 2))
svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
lsa = make_pipeline(text_vectorizer, svd)

clf = SVC(C=150, gamma=2e-2, probability=True)
prediction_pipeline = make_pipeline(lsa, clf)
prediction_pipeline.fit(df_train["data"], df_train["target"])
prediction_pipeline.score(df_test["data"], df_test["target"])
