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


def import_reddit_set(year=2019, rows=999999):
    """
    Get the set of reddit comments uploaded by Alex Cave
    :return:
        dataframe of 'rows' rows of the dataset containing the body of the comment and date fields
    """
    client = bigquery.Client()
    sql = (
        "SELECT body, TIMESTAMP_SECONDS(created_utc) as date FROM `goldenfleece.final_task.ps_reddit_comments_uk_{}` LIMIT {}"
    ).format(year, rows)

    return client.query(sql).to_dataframe()


def import_uk_confidence():
    all_confidence = pd.read_csv('data/consumer_confidence_index.csv',
                               usecols=['TIME', 'Value', 'LOCATION'])

    uk_confidence = all_confidence.loc[all_confidence['LOCATION'] == "GBR"]

    assert all(pd.value_counts(uk_confidence['TIME'] == 1)), "duplicate entries for the same time period"

    date = pd.to_datetime(uk_confidence['TIME'], format="%Y-%m")


    # clean dataframe:
    df = pd.DataFrame({'date': date, 'confidence': uk_confidence.Value})
    df['date'] = df['date'].dt.to_period("M")
    df = df.set_index(['date'], drop=True)

    return df


def replace_quarterly(row):
    if row["Title"].endswith("Q1"):
        return row["Title"].replace(" Q1", "-01-01")
    elif row["Title"].endswith("Q2"):
        return row["Title"].replace(" Q2", "-04-01")
    elif row["Title"].endswith("Q3"):
        return row["Title"].replace(" Q3", "-07-01")
    elif row["Title"].endswith("Q4"):
        return row["Title"].replace(" Q4", "-10-01")


def import_household_savings() -> pd.DataFrame:
    """
    Returns UK household savings ratios
    """
    household_savings_df = pd.read_csv("data/household_savings_ratio.csv")
    household_savings_df["quarterly_data"] = [
        True if "Q" in x else False for x in household_savings_df["Title"]
    ]
    household_savings_df = household_savings_df[
        household_savings_df["quarterly_data"] == True
    ]
    household_savings_df["date"] = pd.to_datetime(
        household_savings_df.apply(replace_quarterly, axis=1)
    ).dt.to_period("M")
    household_savings_df = household_savings_df.set_index("date").resample("M").ffill()
    household_savings_df = household_savings_df.rename(
        {
            "Households (S.14): Households' saving ratio (per cent): Current price: £m: SA": "savings_ratio"
        },
        axis=1,
    )
    return household_savings_df[["savings_ratio"]]
