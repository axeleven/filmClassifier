import text_processing
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None)     # Affiche tout le contenu d'une cellule, même s'il est long
pd.set_option('display.expand_frame_repr', False) 
dataset = pd.read_csv("dataset/Trending_Movies.csv")
dataset = dataset.dropna(subset=['vote_average', 'overview'])
vote_average = round(dataset['vote_average'])
texts = dataset['overview'].to_list()
X_train, X_test, y_train, y_test = train_test_split(
    texts, vote_average, test_size=0.2, random_state=42, shuffle=True
)

rel_extractor = text_processing.RelationTfidfExtractor()
rel_extractor.fit(X_train)
X_train_features = text_processing.extract_features(X_train, rel_extractor)
X_test_features = text_processing.extract_features(X_test, rel_extractor)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)