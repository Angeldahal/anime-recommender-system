from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
import streamlit as st
import joblib
import os


# Load the data
data_path = './Data/processed_anime.csv'

# load the required vectorizer and indices if saved before
tfv_file = 'tfv.joblib'
sig_file = 'sig.joblib'
indices_file = 'indices.joblib'

@st.cache_data
def load_csv():
    anime_data = pd.read_csv(data_path)
    return anime_data

anime_data = load_csv()

if os.path.exists(tfv_file) and os.path.exists(sig_file) and os.path.exists(indices_file):
    print("Found existing joblib files, loading them... ")
    tfv_matrix = joblib.load(tfv_file)
    sig = joblib.load(sig_file)
    indices = joblib.load(indices_file)
    print("Files loaded successfully!")
else:
    print("Files Not found, generating them... ")
    genres_str = anime_data['Genres'].str.split(',').astype(str)

    #initialize tfidfvectorizer
    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words='english')

    #Use TfidfVectorizer to transform the genres_str into a sparse matrix
    tfv_matrix = tfv.fit_transform(genres_str)

    #compute the sigmoid kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

    # Create a Pandas Series object where the index is the anime names and the values are the indices in anime_data
    indices = pd.Series(anime_data.index, index=anime_data["Name"])
    indices = indices.drop_duplicates()

    joblib.dump(tfv_matrix, tfv_file)
    joblib.dump(sig, sig_file)
    joblib.dump(indices, indices_file)
    print("Files generated successfully!")

def give_rec(title, sig=sig):
    """
    Recommends similar anime based on the give title.

    Parameters:
    - title (str): The title of the anime for which recommendations are requested.
    - sig (numpy.ndarray): The similarity matrix between anime titles. Defaults to sig.

    Returns:
    - top_anime (pandas.DataFrame): A Dataframe containing the top recommended anime titles along with their ratings.

    Usage Example:
    >>> give_rec('Attack on Titan')
    """

    idx = indices[title]

    sig_scores = list(enumerate(sig[idx]))

    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    anime_indices = [i[0] for i in sig_scores[1:11]]

    top_anime = pd.DataFrame({
        "Anime name": anime_data['Name'].iloc[anime_indices].values,
        "Rating": anime_data['Score'].iloc[anime_indices].values,
    })

    return top_anime


# Set up the Streamlit App

st.title("Anime Recommender System")
options = anime_data['Name'].tolist()
options.append('Type name here ...')

user_input = st.selectbox(
    'Enter the name of an anime you like:', options=options, index=len(options)-1
)

if user_input == options[-1]:
    pass
elif user_input:
    try:
        recommendations = give_rec(user_input)
        st.write(f"Recommended anime similar to {user_input}:")
        st.table(recommendations)
    except KeyError:
        st.write(f"Sorry, {user_input} is not in our database.")