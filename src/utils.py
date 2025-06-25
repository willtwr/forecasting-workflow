import pandas as pd


def sort_column_by_keywords(dataframe: pd.DataFrame, keywords: list[str], first: bool = True):
    if first:
        new_dataframe = dataframe[[c for c in dataframe if c in keywords] + [c for c in dataframe if c not in keywords]]
    else:
        new_dataframe = dataframe[[c for c in dataframe if c not in keywords] + [c for c in dataframe if c in keywords]]

    return new_dataframe
