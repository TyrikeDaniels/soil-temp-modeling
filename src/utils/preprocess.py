from typing import Union
import pandas as pd


def preprocess(season: Union[str, None] = None) -> pd.DataFrame:
    """
    Load and preprocess raw soil temperature data without scaling.

    Args:
        season (str or None): Optional season filter ('winter' or 'summer'). Defaults to None (all data).

    Returns:
        pd.DataFrame: The raw dataset filtered by season if specified.
    """
    df = pd.read_csv("../data/Grand Forks_daily updated.csv")
    df['Time(CST)'] = pd.to_datetime(df['Time(CST)'], format='%m/%d/%Y')
    df['Month'] = df['Time(CST)'].dt.month
    df['Year'] = df['Time(CST)'].dt.year
    df['day'] = df['Time(CST)'].dt.dayofyear
    df.drop(columns=['Time(CST)'], inplace=True)

    if season == 'winter':
        df = df[(df['Month'] >= 11) | (df['Month'] <= 3)]
    elif season == 'summer':
        df = df[(df['Month'] >= 6) & (df['Month'] <= 9)]

    return df
