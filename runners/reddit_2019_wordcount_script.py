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
word_frequency_pipe = Pipeline([
    ('remove_non_ascii', RemoveNonAscii()),
    ('remove_punctuation', RemovePunctuation()),
    ('lemmatize', WordLemmatizer()),
    ('count_vec', CountVectorizer(stop_words='english',
                                  lowercase=True,
                                  #preprocessor=None,  # already done above
                                  #tokenizer=None,  # nltk_word_tokenizer works here, but let's try without first.
                                  ngram_range=(1, 3), # this is very memory expensive!
                                  #max_features=2000, # this stops it crashing with memory error..
                                  #vocabulary= , # helpful, we should enter alex's vocab here!
                                  #binary=False,
                                  #encoding='ascii',
                                  #strip_accents=None)
                                    ))
                                ])

word_tfidf_pipe = Pipeline([
    ('remove_non_ascii', RemoveNonAscii()),
    ('remove_punctuation', RemovePunctuation()),
    ('lemmatize', WordLemmatizer()),
    ('count_vec', CountVectorizer(stop_words='english',
                                  lowercase=True,
                                  max_features=5000)),
     ('tfidf', TfidfTransformer()),
                                ])



reddit = import_reddit_set(rows=999999)
reddit['date'] = reddit['date'].dt.to_period("M")
reddit.head()

reddit.set_index('date', inplace=True)

reddit.head()
pd.unique(reddit.index)


months_of_text = []
index = []
for month in pd.unique(reddit.index):

    month_of_text = ''.join(reddit.loc[month, 'body'].values.tolist())

    # remove digits..
    month_of_text = ''.join(i for i in month_of_text if not i.isdigit())

    months_of_text.append(month_of_text)
    index.append(month)


# For demonstration purposes..
transformer = CountVectorizer()
matrix = transformer.fit_transform(months_of_text)
result = pd.DataFrame(matrix.toarray(), columns= transformer.get_feature_names(), index=index)
result.head(12)

"""
# Now have word vectors across reddit for each month!
To do: use alex's list of words as the vocabulary for the CountVectorizer

**This is located under BigQuery -> GoldenFleece -> finaltask -> alltones**

word_frequency_pipe = Pipeline([
    ('remove_non_ascii', RemoveNonAscii()),
    ('remove_punctuation', RemovePunctuation()),
    ('lemmatize', WordLemmatizer()),
    ('count_vec', CountVectorizer(stop_words='english',
                                  lowercase=True
                                  vocabulary= IMPORTED_ALLTONES WORDS))
                                ])


"""

## GCP is refusing to run any intensive code right now.
#matrix_full_pipeline = word_frequency_pipe.fit_transform(months_of_text)

#matrix_tfidf = word_tfidf_pipe.fit_transform(months_of_text)
