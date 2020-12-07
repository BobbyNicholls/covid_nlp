"""
Retrieve word frequencies from the reddit text
"""
import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *


from utils.text_analysis_transformers import RemovePunctuation, RemoveNonAscii
from utils.text_analysis_transformers import NltkWordTokenizer, WordLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from utils.TimeBasedCV import TimeBasedCV

from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

from sklearn.svm import SVR


# @todo: look into hashing for speed, and the below vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

word_frequency_pipe = Pipeline([
    ('remove_non_ascii', RemoveNonAscii()),
    ('remove_punctuation', RemovePunctuation()),
    ('lemmatize', WordLemmatizer()),
    ('count_vec', CountVectorizer(stop_words='english',
                                  lowercase=True,
                                  preprocessor=None,  # already done above
                                  tokenizer=None,  # nltk_word_tokenizer works here, but let's try without first.
                                  ngram_range=(1, 5),
                                  max_df=0.9,
                                  min_df=0.1,
                                  max_features=None,
                                  vocabulary=None,
                                  binary=False,
                                  encoding='ascii',
                                  strip_accents=None)),
    ])


## Load in the reddit data:
reddit = pd.read_csv('data/reddit10k.csv')
reddit.date = pd.to_datetime(reddit.date)
reddit.set_index('date', inplace=True)

# Ah, awkward. Only one month of data. Will demo with days.
pd.value_counts(reddit.index.month)
reddit['day'] = reddit.index.day

preprocessed = [word_frequency_pipe.fit_transform(reddit.loc[reddit.day == day])
                for day in pd.unique(reddit.index.day)]

# Import the outcome variable and preprocess
def import_uk_confidence():
    all_confidence = pd.read_csv('data/consumer_confidence_index.csv',
                               usecols=['TIME', 'Value', 'LOCATION'])

    uk_confidence = all_confidence.loc[all_confidence.LOCATION == "GBR"]

    assert all(pd.value_counts(uk_confidence.TIME) == 1), "duplicate entries for the same time period"

    date = pd.to_datetime(uk_confidence.TIME, format="%Y-%m")

    # clean dataframe:
    df = pd.DataFrame({'date': date, 'value': uk_confidence.Value})

    return df

y = import_uk_confidence()