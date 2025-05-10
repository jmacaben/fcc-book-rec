# Some of the given code may have been changed for use outside of a notebook environment

# Cell 1 (given)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import zipfile
import requests

# Cell 2 (given)
url = "https://cdn.freecodecamp.org/project-data/books/book-crossings.zip"
zip_path = "book-crossings.zip"

if not os.path.exists(zip_path):
    print("Downloading...")
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print(f"Failed to download: Status code {response.status_code}")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall()

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# Cell 3 (given)
# Import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})
