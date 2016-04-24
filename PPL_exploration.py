import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import scale


def PPL_data():
    df = pd.read_csv('data/raw/PPL_reservationdata.csv', encoding='ISO-8859-1')
    df = df.dropna()
    df.columns = [x.strip() for x in df.columns]
    df['traveldist'] = df['traveldist'].astype(float)
    return df

def zips_binary(df):
    """Create a binary representation of the each facility zip code

    Parameters
    ----------
    df : pd.DataFrame
        reservation-level

    Returns
    -------
    zips_binary : np.ndarray
    """
    df = df.copy()
    zips = df[['Park', 'FacZip']]
    zips = zips.drop_duplicates()
    zips_binary = pd.get_dummies(zips[['FacZip']])
    return zips_binary.values

def get_numeric_data(df, by='Park', cols=['NumPeople', 'leadtime',
                                          'duration', 'traveldist']):
    """Standardize `cols` in the `by`-level representation of the data
    This representation is based on the median on the values in `cols`

    Parameters
    ----------
    df : pd.DataFrame
        reservation-level
    by : str or list
        if a single column, use either string or list
        is multiple columns, use a list
    cols : list
        the columns to standardize

    Returns
    -------
    by_cols : pd.DataFrame
        non-standardized representation
    values : np.ndarray
        standardized representation
    """
    by_cols = df.groupby(by)[cols].agg('median').reset_index()
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

def facility_description():
    return NotImplemented

def knn_recommender(k=5, scores):
    knn = NearestNeighbors(n_neighbors=k).fit(scores)
    _, indices = knn.kneighbors(scores)
    return indices
