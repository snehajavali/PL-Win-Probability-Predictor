# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        batting_team  = request.form["batting_team"]
        bowling_team  = request.form["bowling_team"]
        city          = request.form["city"]
        runs          = float(request.form["runs"])
        overs         = float(request.form["overs"])
        wickets       = float(request.form["wickets"])

        values = pd.DataFrame([[
            batting_team, bowling_team, city, runs, overs, wickets
        ]], columns=["batting_team","bowling_team","city","runs","overs","wickets"])

        winner = model.predict(values)[0]
        result = f"Predicted Winner: {winner}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
