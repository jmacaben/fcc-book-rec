# 📚🔍Book Recommendation Engine using KNN
A book recommendation project built with scikit-learn, pandas, and SciPy that uses K-Nearest Neighbors (KNN) to develop a model that finds similar books based on user ratings. Specifically, the program returns a list of five recommended books for a given title.


> 🧠 **This challenge was provided by [freeCodeCamp’s Machine Learning with Python course](https://www.freecodecamp.org/learn/machine-learning-with-python/).**


## 🛠 What I Did

- Filtered the Book-Crossings dataset, which contains 1.1 million ratings of 270,000 books by 90,000 users by removing...
  - Users with less than 200 ratings
  - Books with less than 100 ratings
  - Duplicate ratings
- Pivoted the dataframe and converted it into a sparse matrix
- Fit a KNN model to compare the similarity between books based on users' ratings


## 🤔 What I Learned

- **Data cleaning**: By removing users with fewer than 200 ratings and books with fewer than 100 ratings, I reduced noise in the dataset, ensuring that the recommendations were based on reliable data. Additionally, handling duplicate ratings helped eliminate redundant information.
- **Using a sparse matrix:** I used `csr_matrix` to convert the ratings dataframe into a sparse matrix format. CSR (Compressed Sparse Row) helps save memory by storing only the non-zero elements, i.e. a value of 0 in the matrix is where the user didn't rate a book.
- **Initializing a KNN model**: The model uses cosine similarity, a metric that measures the similarity between two vectors based on the angle between them. In this case, it calculates how similar the ratings of two books are by comparing users' preferences. Also, the brute-force approach is used to calculate the nearest neighbors, which is efficient for smaller datasets.


## 🚀 Future Improvements

- **Content-Based Filtering**: This recommendation system is based purely on collaborative filtering, not content. That means the model doesn’t analyze what a book is about — it only looks at patterns in user ratings. For example, if several users rated 'Twilight' and 'Interview with the Vampire' highly, the model infers they are similar, even though it doesn't "know" they're both vampire novels. The recommendations are driven by user behavior, not book descriptions, genres, or themes. Adding data like genre, author, or keywords could help the model make smarter recommendations, especially for lesser-known books with fewer ratings.





