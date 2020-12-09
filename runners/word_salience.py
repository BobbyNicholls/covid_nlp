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

from utils.data_utils import import_reddit_set

def import_uk_confidence():
    all_confidence = pd.read_csv('data/consumer_confidence_index.csv',
                               usecols=['TIME', 'Value', 'LOCATION'])

    uk_confidence = all_confidence.loc[all_confidence["LOCATION"] == "GBR"]

    assert all(pd.value_counts(uk_confidence["TIME"]) == 1), "duplicate entries for the same time period"

    date = pd.to_datetime(uk_confidence["TIME"], format="%Y-%m")

    # clean dataframe:
    df = pd.DataFrame({'date': date, 'value': uk_confidence["Value"]})

    return df

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

reddit = import_reddit_set(rows=999999)
reddit.describe()
reddit.head()


# In[7]:


reddit["date"] = pd.to_datetime(reddit["date"])
reddit.set_index('date', inplace=True)
reddit.head()


# In[8]:


reddit['month'] = reddit.index.month
reddit.head()


# # But this doesn't look quite right!

# In[12]:


pd.unique(reddit.index.month)


# In[16]:


reddit.iloc[70000, :]


# ## Outcome variable: 

# In[10]:


# Import the outcome variable and preprocess
uk_confidence = import_uk_confidence()

uk_confidence.head()

