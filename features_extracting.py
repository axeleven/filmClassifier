import text_processing
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None)     # Affiche tout le contenu d'une cellule, même s'il est long
pd.set_option('display.expand_frame_repr', False) 

dataset = pd.read_csv("dataset/Trending_Movies.csv")
dataset = dataset[dataset['vote_count'] != 0]
dataset = dataset.dropna(subset=['vote_average', 'overview', 'release_date'])
dataset['year'] = pd.to_datetime(dataset['release_date'], errors='coerce').dt.year
vote_average = round(dataset['vote_average'])
texts = list(zip(dataset['overview'].to_list(), dataset['year']))
vote_average = round(dataset['vote_average'])

X_train, X_test, y_train, y_test = train_test_split(
    texts, vote_average, test_size=0.2, random_state=42, shuffle=True, stratify=vote_average
)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print("SAVED")
rel_extractor = text_processing.RelationTfidfExtractor()
rel_extractor.fit(X_train)
X_train_features = text_processing.extract_features(X_train, rel_extractor)
X_test_features = text_processing.extract_features(X_test, rel_extractor)

X_train_df = pd.DataFrame(X_train_features).fillna(0)
X_test_df = pd.DataFrame(X_test_features).fillna(0)
columns = X_train_df.columns
X_test_df = X_test_df.reindex(columns=columns, fill_value=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

np.save("X_train_scaled.npy", X_train_scaled)
np.save("X_test_scaled.npy", X_test_scaled)


