"""
For data import and manipulation, for example BQ API functions should be written here.
"""

import pandas as pd

try:
    from google.cloud import bigquery
except ModuleNotFoundError:
    print("gcp not installed")


def import_toy_set():
    """
    Just a function that will bring in a very small set for PoC purposes
    :return: dataframe with just URLS and text snippets
    """
    raw_text_df = pd.read_csv("data/text_files-2020-11-20.csv")
    raw_text_df.columns = [x.lower() for x in raw_text_df.columns]
    return raw_text_df


def import_5k_covid_toy_set():
    """
    Just a function that will bring in a very small set for PoC purposes
    :return: dataframe with just URLS and text snippets
    """
    raw_text_df = pd.read_csv("data/covid_5k_snippets-20201120-21.csv")
    raw_text_df.columns = [x.lower() for x in raw_text_df.columns]
    return raw_text_df


def import_reddit10k():
    """
    Just a function that will bring in a very small set for PoC purposes
    :return: dataframe with just URLS and text snippets
    """
    raw_text_df = pd.read_csv("data/reddit10k.csv")
    raw_text_df.columns = [x.lower() for x in raw_text_df.columns]
    return raw_text_df


def import_reddit_set(rows):
    """
    Get the set of reddit comments uploaded by Alex Cave
    :return:
        dataframe of 'rows' rows of the dataset containing the body of the comment and date fields
    """
    client = bigquery.Client()
    sql = (
        "SELECT body, date FROM `goldenfleece.vaccine_poc.ps_reddit_comments_vaccine` LIMIT {}"
    ).format(rows)

    return client.query(sql).to_dataframe()
