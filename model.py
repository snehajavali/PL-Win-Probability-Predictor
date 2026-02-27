import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load dataset
df = pd.read_csv("ipl_matches.csv")

features = df[["batting_team","bowling_team","city","runs","overs","wickets"]]
target = df["match_winner"]

cat_features = ["batting_team","bowling_team","city"]

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

accuracy = model.score(X_test, y_test)
print(f"Model training completed!")
print(f"Model Accuracy: {round(accuracy*100,2)}%")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
