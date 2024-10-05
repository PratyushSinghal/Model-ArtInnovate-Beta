import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump

df = pd.read_csv("Link to our dataset")

# One-hot encode the 'movement' and 'period' columns
movement_df = pd.get_dummies(df['movement'])
period_df = pd.get_dummies(df['period'])
movement_df.drop(columns='[nan]', inplace=True)
period_df.drop(columns='[nan]', inplace=True)
df.drop(columns=['movement', 'period'], inplace=True)
df = df.join(movement_df)
df = df.join(period_df)

df['yearCreation'] = pd.to_numeric(df['yearCreation'], errors="coerce")
df['yearCreation'].fillna(df['yearCreation'].median(), inplace=True)

df['price'] = df['price'].str.replace("₹", "")
df['price'] = df['price'].str.replace(".", "")
df['price'] = pd.to_numeric(df['price'])

# Vectorize the 'condition' column
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
condition = vectorizer.fit_transform(df['condition'])
condition_labels = pd.DataFrame(condition.toarray(), columns=vectorizer.get_feature_names_out())

df.drop(columns=['condition', 'artist'], inplace=True)
condition_labels = condition_labels.astype(int)
df = df.join(condition_labels)

df = df.drop(df.columns[[0, 1, 2, 5]], axis=1)

X = df.loc[:, ~df.columns.isin(['artist', 'title', 'condition', 'title_left', 'price'])]
Y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

regr = RandomForestRegressor(max_depth=20, random_state=0)
regr.fit(X_train, y_train)

# Evaluation
print("Training Score: ", regr.score(X_train, y_train))
print("Testing Score: ", regr.score(X_test, y_test))
print("Testing Mean Absolute Error: ", mean_absolute_error(y_test, regr.predict(X_test)))

# Feature importance analysis
importances = regr.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
feat_labels = df.columns[1:]

for f in range(min(40, len(feat_labels))):
    print("%2d) %-*s %f" % (f + 1, 10, feat_labels[sorted_indices[f]], importances[sorted_indices[f]]),
          "index: ", df.columns.get_loc(str(feat_labels[sorted_indices[f]])))

dump(regr, 'Categorical.joblib')
