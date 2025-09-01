'''
Module for preprocessing news data for sentiment analysis.
'''

import pandas as pd
from google import genai
from google.genai import types
import json

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

def clean_data(df):
    """
    Clean the news data for financial RoBERTa model.
    Expected cleaning steps:
    - Removing extra whitespace
    - Normalize Unicode characters
    Parameters:
    df (pd.DataFrame): DataFrame containing the news data.
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df['body'] = df['body'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['body'] = df['body'].apply(lambda x: x.encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
    return df

def custom_ticker_aliases(tickers: list, gemini_api_key: str) -> list:
    '''
    Custom function to process tickers using Gemini API.
    Gemini will be used to get all known aliases for a given ticker.
    For example, AAPL will return ['AAPL', 'Apple Inc', 'Apple', 'Apple Corporation']
    This will help in identifying mentions of the company in the news articles.

    Parameters:
    tickers (list): List of ticker symbols.
    gemini_api_key (str): API key for Gemini.
    Returns:
    list: List of ticker symbols with their aliases.
    '''
    names = []
    client = genai.Client(api_key=gemini_api_key)
    for ticker in tickers:
        prompt = f"Provide a comprehensive list of all known aliases for the ticker symbol '{ticker}'. Format the response as a Python list. For example, AAPL may return ['AAPL', 'Apple Inc', 'Apple', 'Apple Corporation']. Try to include all possible variationsof tickers, full name, short name, corporate forms, etc."
        response = client.models.generate_content(
            model = "gemini-2.5-flash",
            contents = prompt
        )
        text = response.text or ""
        try:
            alias_list = eval(text) if text.strip().startswith("[") else json.loads(text)
            if not isinstance(alias_list, list):
                raise ValueError("Not a list")
        except Exception:
            # fallback: split by commas if parsing fails
            alias_list = [alias.strip().strip('"').strip("'") for alias in text.strip("[]").split(",") if alias.strip()]

        names.extend([ticker]+alias_list)

    unique_names = list(set(names))
    # Strip\n and extra spaces
    unique_names = [name.replace("\n", " ").strip() for name in unique_names if name.strip()]

    return unique_names

def preprocessing_pipeline(df, word_count_threshold=5, gemini_api_key=None, tickers=None):
    """
    Complete preprocessing pipeline for news data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the news data.
    word_count_threshold (int): Minimum number of words required to keep an article.
    gemini_api_key (str): API key for Gemini (optional).
    tickers (list): List of ticker symbols (optional).

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    list: List of ticker symbols with their aliases (if tickers and gemini_api_key are provided).
    """
    df = filter_data(df, word_count_threshold)
    df = clean_data(df)

    ticker_aliases = None
    if gemini_api_key and tickers:
        ticker_aliases = custom_ticker_aliases(tickers, gemini_api_key)

    return df, ticker_aliases