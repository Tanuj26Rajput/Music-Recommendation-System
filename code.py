import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

music_df = pd.read_csv("C:\\Users\\Tanuj Rajput\\OneDrive\\Desktop\\tanujkaggle\\7\\dataset.csv")

numeric_cols = music_df.select_dtypes(include=[np.number]).columns
numeric_cols = numeric_cols.drop(['Unnamed: 0','popularity', 'duration_ms', 'key', 'mode', 'time_signature'])
X = music_df[numeric_cols]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.cluster import KMeans

options = range(2,11)
inertias = []
for n_clusters in options:
    model = KMeans(n_clusters, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

optimal_inertia = 4
kmeans = KMeans(n_clusters=optimal_inertia, random_state=42, n_init=10)
music_df['cluster'] = kmeans.fit_predict(X_scaled)

def recommend_songs(song_name, df, num_recommendations=5):
    song = df[df['track_name'].str.lower() == song_name.lower()]
    if song.empty:
        return "Song not found!"
    song_cluster = song['cluster'].values[0]
    recommendations = df[df['cluster'] == song_cluster].sample(num_recommendations)
    return recommendations[['track_name', 'artists']]

song_input = input("Enter name of the song: ")
recommended_songs = recommend_songs(song_input, music_df)

if isinstance(recommended_songs, str):  
    print(recommended_songs)  
else:
    print("\nRecommended Songs:")
    print("\n{:<70} {:<50}".format("Track Name", "Artist Name"))
    print("=" * 110)
    for _, row in recommended_songs.iterrows():
        print("{:<70} {:<50}".format(row['track_name'], row['artists']))
