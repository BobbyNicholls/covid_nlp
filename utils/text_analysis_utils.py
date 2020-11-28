"""
Module for functions that analyse text
"""

import pandas as pd
from utils.text_preprocessing_utils import normalise, tokenise


def get_cluster(row, kmeans):
    return kmeans.predict(row.values.reshape(1, -1))[0]


def get_common_words(string):
    text_df = pd.DataFrame(normalise(tokenise(string)))
    text_df["count"] = 1
    grouped_df = text_df.groupby([0]).count()
    grouped_df = grouped_df.sort_values(["count"], ascending=False)
    grouped_df = grouped_df[:20]
    grouped_df.plot.bar()
