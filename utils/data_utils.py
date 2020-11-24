"""
For data import and manipulation, for example BQ API functions should be written here.
"""

import pandas as pd


def import_toy_set():
    """
    Just a function that will bring in a very small set for PoC purposes
    :return: dataframe with just URLS and text snippets
    """
    raw_text_df = pd.read_csv("data/text_files-2020-11-20.csv")
    raw_text_df.columns = [x.lower() for x in raw_text_df.columns]
    return raw_text_df
