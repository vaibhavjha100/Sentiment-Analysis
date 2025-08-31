'''
Module for preprocessing news data for sentiment analysis.
'''

import pandas as pd

def load_data(file_path):
    """
    Load news data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: DataFrame containing the news data.
    """
    return pd.read_csv(file_path)

def filter_data(df, word_count_threshold=5):
    """
    Filter out articles with fewer than a specified number of words.

    Parameters:
    df (pd.DataFrame): DataFrame containing the news data.
    word_count_threshold (int): Minimum number of words required to keep an article.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    df['word_count'] = df['body'].apply(lambda x: len(str(x).split()))
    return df[df['word_count'] >= word_count_threshold].drop(columns=['word_count'])