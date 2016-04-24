import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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

def get_cleaned_matrix(cos_scores):
    for i in range(len(cos_scores)):
        for j in range(len(cos_scores)):
            if cos_scores[i][j] < 0.7:
                cos_scores[i][j] = 0
    return cos_scores
