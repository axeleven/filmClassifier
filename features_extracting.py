import text_processing
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


dataset = pd.read_csv("dataset/Trending_Movies.csv")
dataset = dataset[dataset['vote_count'] != 0]
dataset = dataset.dropna(subset=['vote_average', 'overview', 'release_date'])
dataset['year'] = pd.to_datetime(dataset['release_date'], errors='coerce').dt.year.astype(int)
vote_average = round(dataset['vote_average'])
tfid = TfidfVectorizer()
texts = dataset['overview']
vote_average = round(dataset['vote_average'])

X_train, X_test, y_train, y_test = train_test_split(
    texts, vote_average, test_size=0.2, random_state=42, shuffle=True, stratify=vote_average
)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print("SAVED")

rel_extractor = text_processing.RelationTfidfExtractor()
#rel_extractor.fit(X_train)
X_train_features = text_processing.extract_features(X_train, rel_extractor)
X_test_features = text_processing.extract_features(X_test, rel_extractor)

X_train_df1 = pd.DataFrame(X_train_features).fillna(0)
X_test_df1 =pd.DataFrame(X_test_features).fillna(0)
columns = X_train_df1.columns
X_test_df = X_test_df1.reindex(columns=columns, fill_value=0)

X_train_df2 = pd.DataFrame(tfid.fit_transform(X_train).toarray(), columns=tfid.get_feature_names_out())
X_test_df2 = pd.DataFrame(tfid.transform(X_test).toarray(), columns=tfid.get_feature_names_out())

X_train_df = pd.concat([X_train_df1, X_train_df2], axis=1)
X_test_df = pd.concat([X_test_df1, X_test_df2], axis=1)
X_train_year = pd.DataFrame(X_train.reset_index(drop=True)).assign(year=X_train.index.map(lambda idx: dataset.loc[idx, 'year'])).year
X_train_year = pd.DataFrame(X_train_year)
print(X_train_year)
X_test_year = pd.DataFrame(X_test.reset_index(drop=True)).assign(year=X_test.index.map(lambda idx: dataset.loc[idx, 'year'])).year
X_test_year = pd.DataFrame(X_test_year)
X_train_df = pd.concat([X_train_df, X_train_year], axis=1)
X_test_df = pd.concat([X_test_df, X_test_year], axis=1)
print(X_train_df.shape, X_test_df.shape)
print(X_train_df.head())
categorical_cols = X_train_df.select_dtypes(include=['O']).columns
print("Categorical columns:", list(categorical_cols))
svd = TruncatedSVD(n_components=1000, random_state=42)
X_train_svd = svd.fit_transform(X_train_df)
X_test_svd = svd.transform(X_test_df)

X_train_df = pd.DataFrame(X_train_svd)

X_test_df = pd.DataFrame(X_test_svd)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df.to_numpy())
X_test_scaled = scaler.transform(X_test_df.to_numpy())


np.save("X_train_scaled.npy", X_train_scaled)
np.save("X_test_scaled.npy", X_test_scaled)


