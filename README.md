# 🎵 Music Recommendation System using Clustering

This project is an unsupervised learning-based music recommendation system that groups songs using **KMeans clustering** and recommends tracks similar to the input song based on audio features.

## 📌 How It Works
- Extracts numeric features from a music dataset (excluding less relevant ones)
- Standardizes the features using `StandardScaler`
- Applies KMeans clustering to group similar songs
- When a user inputs a song name, the system finds other songs in the same cluster and recommends a few

## 🔧 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib & Seaborn (for visualization)
- KMeans Clustering

## 🚀 Features
- Cluster-based song recommendations
- Clean CLI-based interface
- Fast and efficient for medium-sized datasets