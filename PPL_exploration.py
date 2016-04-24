import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
        each row representing a reservation

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
    numeric : pd.DataFrame
        the standardized representation
    """
    park_num = df.groupby(by)[cols].agg('median').reset_index()
    names = park_num[by].values
    values = scale(park_num[cols])
    numeric = pd.DataFrame(np.column_stack((names, values)), columns=park_num.columns)
    return numeric

def get_distance_matrix(df):
    vs = df[["NumPeople", "leadtime", "duration", "traveldist"]].values
    cos_scores = cosine_similarity(vs, vs)
    return cos_scores

def get_cleaned_matrix(cos_scores):
    for i in range(len(cos_scores)):
        for j in range(len(cos_scores)):
            if cos_scores[i][j] < 0.7:
                cos_scores[i][j] = 0
    return cos_scores
