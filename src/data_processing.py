import pandas as pd

def load_movies(path="data/movies.csv"):
    """Load the movies CSV file"""
    df = pd.read_csv(path)
    return df

def clean_movies(df):
    """Basic cleaning: remove duplicates, fill NaN, etc."""
    df = df.drop_duplicates(subset="title", keep="first")
    df = df.fillna("")
    return df

if __name__ == "__main__":
    df = clean_movies(load_movies())
    print(df.head())
