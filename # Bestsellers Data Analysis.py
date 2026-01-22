# Bestsellers Data Analysis
# My First Data Science Project 🚀

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("bestsellers with categories.csv")

# 1. Explore the data
print("First 5 rows of the dataset:")
print(df.head())

print("\nShape of the dataset:", df.shape)
print("\nColumn names:", df.columns)
print("\nSummary statistics:")
print(df.describe())

# 2. Clean the data
# Remove duplicates
df.drop_duplicates(inplace=True)

# Rename columns for clarity
df.rename(columns={
    'Name': 'Title',
    'Year': 'Publication Year',
    'User Rating': 'Ratings',
    'Reviews': 'Book Review',
    'Price': 'Book Price'
}, inplace=True)

# Ensure correct data type
df['Book Price'] = df['Book Price'].astype(float)

print("\nColumns after renaming:", df.columns)

# 3. Analysis
# Count books per author
author_count = df['Author'].value_counts()
print("\nCount of books per author:\n", author_count)

# Author with most books
top_author = author_count.idxmax()
print("\nAuthor with most books:", top_author)

# Average rating by genre
avg_rating_by_genre = df.groupby('Genre')['Ratings'].mean()
print("\nAverage rating by genre:\n", avg_rating_by_genre)

# Books with high ratings
high_rated = df[df['Ratings'] >= 4.5]
print("\nBooks with ratings >= 4.5:\n", high_rated[['Title', 'Author', 'Ratings']])

# Most expensive book
print("\nMost expensive book:\n", df.loc[df['Book Price'].idxmax()])

# Book with most reviews
print("\nBook with most reviews:\n", df.loc[df['Book Review'].idxmax()])

# 4. Visualizations
# Top 10 authors by number of books
author_count.head(10).plot(kind='bar', title="Top 10 Authors by Number of Books", figsize=(8,4))
plt.ylabel("Number of Books")
plt.show()

# Average ratings by genre
avg_rating_by_genre.plot(kind='bar', title="Average Ratings by Genre", figsize=(6,4))
plt.ylabel("Average Rating")
plt.show()
