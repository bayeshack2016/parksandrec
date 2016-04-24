import re

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale


def PPL_data():
    df = pd.read_csv('data/raw/PPL_reservationdata.csv', encoding='ISO-8859-1')
    df = df.dropna()
    df.columns = [x.strip() for x in df.columns]
    df['traveldist'] = df['traveldist'].astype(float)
    df['StartMonth'] = df['StartDate'].str.split('-').str[1]
    return df

def binarize(df, by='Park', col='FacZip', no_duplicates=True, as_df=False):
    """Create a binary representation at the `by`-level for `cols`

    Parameters
    ----------
    df : pd.DataFrame
        reservation-level
    by : str
        representation-level
    col : str
        feature to binarize
    no_duplicates : bool
        remove duplicates
    as_df : bool
        represent as pd.DataFrame with `by` column

    Returns
    -------
    cols_binary : np.ndarray or pd.DataFrame
    """
    df = df.copy()
    cols = df[[by, col]]
    if no_duplicates:
        cols = cols.drop_duplicates()
    cols_binary = pd.get_dummies(cols[[col]])
    if as_df:
        cols_binary[by] = df[by]
        return cols_binary
    else:
        return cols_binary.values

def get_numeric_data(df, by='Park', cols=['NumPeople', 'leadtime',
                                          'duration', 'traveldist'],
                     operation='median'):
    """Standardize `cols` in the `by`-level representation of the data
    This representation is based on the `operation` on the values in `cols`

    Parameters
    ----------
    df : pd.DataFrame
        reservation-level
    by : str or list
        if a single column, use either string or list
        is multiple columns, use a list
    cols : list
        the columns to standardize
    operation : str
        default median

    Returns
    -------
    by_cols : pd.DataFrame
        non-standardized representation
    values : np.ndarray
        standardized representation
    """
    by_cols = df.groupby(by)[cols].agg(operation).reset_index()
    values = scale(by_cols[cols])
    return by_cols, values

def similarity_matrix(M, threshold=0.7):
    """Calculate the cosine similarity matrix and
    set values below `threshold` to 0.0

    Parameters
    ----------
    M : np.ndarray
        shape (n_samples, n_features)
    threshold : float
        the value under which to set cell values to 0.0

    Returns
    -------
    cos_scores : np.ndarray
    """
    cos_scores = cosine_similarity(M, M)
    cos_scores[cos_scores<threshold] = 0.0
    return cos_scores

def knn_recommender(scores, k=5):
    knn = NearestNeighbors(n_neighbors=k).fit(scores)
    _, indices = knn.kneighbors(scores)
    return indices

def remove_tags(text, tags=['h2', 'br', 'h4']):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(tags)]
    return re.sub('\.', '. ', soup.text)

def split_on_sentence(text, first_sentence=True):
    text = remove_tags(text)
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    if first_sentence:
        return sent_tokenizer.tokenize(text)[0]
    else:
        return sent_tokenizer.tokenize(text)

def facility_description(df, facility_name):
    text = df[df.FACILITYNAME == facility_name].FACILITYDESCRIPTION.values[0]
    return split_on_sentence(text)
