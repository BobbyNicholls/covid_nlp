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

from utils.data_utils import import_reddit_set, import_uk_confidence


from sklearn.feature_extraction.text import TfidfTransformer

from google.cloud import bigquery


def get_tone_vocab():
    """
    load the tone based words from Alex's work to use
    as the columns for the word freq counter.
    """
    client = bigquery.Client(location="US", project="goldenfleece")

    query = """
        SELECT *
        FROM final_task.alltones
    """
    query_job = client.query(
        query,
        # Location must match that of the dataset(s) referenced in the query.
        location="US",
        project='goldenfleece'
    )  # API request - starts the query

    all_tones_words = query_job.to_dataframe()
    
    unique_words = all_tones_words.drop_duplicates('words', ignore_index=True)
    unique_words_dict = unique_words['words'].to_dict()
    
    # mapping needs to be reversed for the sklearn countvectorizer!
    vocab = {v: k for k, v in unique_words_dict.items()}
    
    return vocab


vocab = get_tone_vocab()

reddit = import_reddit_set(rows=999999)
reddit['date'] = reddit['date'].dt.to_period("M")
reddit.set_index('date', inplace=True)

reddit.head()


def get_list_of_monthly_text()
    months_of_text = []
    index = []
    for month in pd.unique(reddit.index):

        month_of_text = ''.join(reddit.loc[month, 'body'].values.tolist())

        # remove digits..
        month_of_text = ''.join(i for i in month_of_text if not i.isdigit())

        months_of_text.append(month_of_text)
        index.append(month)
        
    return index, month

index, months_of_text = get_list_of_monthly_text()

word_frequency_pipe = Pipeline([
    ('remove_non_ascii', RemoveNonAscii()),
    ('remove_punctuation', RemovePunctuation()),
    ('lemmatize', WordLemmatizer()),
    ('count_vec', CountVectorizer(stop_words='english',
                                  lowercase=True,
                                  ngram_range=(1, 3), # this is very memory expensive!
                                  vocabulary= vocab)
                                    )
                                ])

word_tfidf_pipe = Pipeline([
    ('remove_non_ascii', RemoveNonAscii()),
    ('remove_punctuation', RemovePunctuation()),
    ('lemmatize', WordLemmatizer()),
    ('count_vec', CountVectorizer(stop_words='english',
                                  lowercase=True,
                                  vocabulary= vocab)),
     ('tfidf', TfidfTransformer()),
                                ])


def get_results(pipeline_transformer: Pipeline, text: list) -> pd.DataFrame:
    matrix = pipeline_transformer.fit_transform(text)
    
    return pd.DataFrame(matrix.toarray(), columns= pipeline_transformer['count_vec'].get_feature_names(), index=index)


word_frequency_result = get_results(word_frequency_pipe, months_of_text)
tfidf_result = get_results(word_tfidf_pipe, months_of_text)

