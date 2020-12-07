"""
Sklearn word preprocessing pipeline for the NLP Vaccine Project
"""

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords

from sklearn.base import TransformerMixin, BaseEstimator

import unicodedata
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re


class NltkWordTokenizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y= None, **fit_params):
        return self

    def transform(self, X, y= None, **fit_params):
        return [word_tokenize(i) for i in X]

    def fit_transform(self, X, y= None):
        return self.transform(X)


class RemovePunctuation(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Remove punctuation from list of tokenized words"""
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        new_words = []
        for word in X:
            new_word = re.sub(r"[^\w\s]", "", word)
            if new_word != "":
                new_words.append(new_word)
        return new_words

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class RemoveNonAscii(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Remove punctuation from list of tokenized words"""
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in X:
            new_word = (
                unicodedata.normalize("NFKD", word)
                    .encode("ascii", "ignore")
                    .decode("utf-8", "ignore")
            )
            new_words.append(new_word)
        return new_words

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class WordLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """Remove punctuation from list of tokenized words"""
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        """Remove non-ASCII characters from list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word, pos="v") for word in X]
        return lemmas

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)
