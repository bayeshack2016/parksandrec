import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_numeric_data():
    df = pd.read_csv("PPL_reservationdata.csv", encoding="ISO-8859-1")
    df['traveldist'] = df['traveldist'].astype(float)
    park_num = df.groupby("Park")[["NumPeople", "leadtime", "duration", "traveldist"]].agg('median').reset_index()
    park_num.leadtime = (park_num.leadtime - park_num.leadtime.mean())/park_num.leadtime.std(ddof=0)
    park_num.duration = (park_num.duration - park_num.duration.mean())/park_num.duration.std(ddof=0)
    park_num.NumPeople = (park_num.NumPeople - park_num.NumPeople.mean())/park_num.NumPeople.std(ddof=0)
    park_num.traveldist = (park_num.traveldist - park_num.traveldist.mean())/park_num.traveldist.std(ddof=0)
    return park_num


def get_disctance_matrix(df):
    vs = df[["NumPeople", "leadtime", "duration", "traveldist"]].values
    cos_scores = cosine_similarity(vs, vs)
    return cos_scores
