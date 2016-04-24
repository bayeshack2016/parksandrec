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

def get_numeric_data(df):
    park_num = df.groupby("Park")[["NumPeople", "leadtime", "duration", "traveldist"]].agg('median').reset_index()
    park_num.leadtime = (park_num.leadtime - park_num.leadtime.mean())/park_num.leadtime.std(ddof=0)
    park_num.duration = (park_num.duration - park_num.duration.mean())/park_num.duration.std(ddof=0)
    park_num.NumPeople = (park_num.NumPeople - park_num.NumPeople.mean())/park_num.NumPeople.std(ddof=0)
    park_num.traveldist = (park_num.traveldist - park_num.traveldist.mean())/park_num.traveldist.std(ddof=0)
    return park_num

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
