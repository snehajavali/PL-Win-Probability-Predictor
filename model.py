# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df = pd.read_csv("data/ipl_matches.csv")

# Simplify features
features = df[[
    "batting_team", "bowling_team", "city", "runs", "overs", "wickets"
]]
target = df["match_winner"]

# Categorical and numeric
cat_features = ["batting_team", "bowling_team", "city"]
num_features = ["runs", "overs", "wickets"]

# Pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=500))
])

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
print("Model training completed!")

joblib.dump(model, "models/model.pkl")
