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

# MY CODE
# Remove users with fewer than 200 ratings and books with fewer than 100 ratings
user_rating_counts = df_ratings['user'].value_counts()
users_to_keep = user_rating_counts[user_rating_counts > 200].index
book_rating_counts = df_ratings['isbn'].value_counts()
books_to_keep = book_rating_counts[book_rating_counts > 100].index

df_ratings = df_ratings[df_ratings['user'].isin(users_to_keep)]
df_ratings = df_ratings[df_ratings['isbn'].isin(books_to_keep)]

# Merge the filtered data
df_merged = pd.merge(right = df_ratings, left = df_books, on = "isbn")

# Remove duplicates, i.e. when a user has rated the same book multiple times
df_cleaned = df_merged.drop_duplicates(["title", "user"])

# Pivot the dataframe to create a matrix where rows are books, columns are users, and values are ratings
df_pivot = df_cleaned.pivot(index = 'title', columns = 'user', values = 'rating').fillna(0)

# Create a sparse matrix from the pivoted dataframe
matrix = csr_matrix(df_pivot.values)

# Develop a model that shows books that are similar to a given book using K-Nearest Neighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(matrix)

# function to return recommended books - this will be tested
def get_recommends(book = ""):
    # Get row
    x = df_pivot.loc[book].array.reshape(1, -1)

    # Use the trained model to find the 6 nearest neighbors (books) to the specified book
    distances, indices = model_knn.kneighbors(x, n_neighbors = 6)

    # Create a list to store the recommended books and their distances
    recs=[]
    for distance, indice in zip(distances[0], indices[0]):
       if distance != 0: # Exclude the input book itself
          rec = df_pivot.index[indice]
          recs.append([rec, distance])

    # Sort the recommended books by distance (similarity) in ascending order     
    recommended_books = [book, recs[::-1]]
    return recommended_books

# Cell 5 (given)
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()