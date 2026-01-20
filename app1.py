
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify
import sqlite3
import os
import pandas as pd


app = Flask(__name__)

# Load model artifacts with basic error handling
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("onehot_encoder.joblib")
except Exception as e:
    model = scaler = encoder = None
    print(f"Error loading model artifacts: {e}")

DB_FILE = "predict.db"

def init_db():
    conn=sqlite3.connect(DB_FILE)
    c=conn.cursor()
    c.execute("""
              CREATE TABLE IF NOT EXISTS predictions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              Age REAL,
              BMI REAL,
              Glucose REAL,
              Gender TEXT,
              Smoker TEXT,
              FamilyHistory TEXT,
              Result TEXT
        )
    """)

    conn.commit()
    conn.close()

@app.route("/", methods=["GET","POST"])
def index():
    result= None
    if request.method  == "POST":
        age=float(request.form["Age"])
        # fix: read BMI and Glucose from correct form fields
        bmi=float(request.form["BMI"])
        glucose=float(request.form["Glucose"])
        x_num=np.array([[age, bmi, glucose]])

        #----get categorial inputs---_#

        gender=request.form["Gender"]
        smoker=request.form["Smoker"]
        family=request.form["FamilyHistory"]
        x_cat=np.array([[gender , smoker, family]])

        #_apply preprocessing#

        # ensure artifacts available
        if None in (model, scaler, encoder):
            result = "Error: model or preprocessing artifacts not loaded. Check server logs."
        else:
            x_num_scaled = scaler.transform(x_num)
            x_cat_encoded = encoder.transform(x_cat)
            try:
                x_cat_arr = x_cat_encoded.toarray() if hasattr(x_cat_encoded, "toarray") else x_cat_encoded
            except Exception:
                x_cat_arr = x_cat_encoded

            x_final = np.hstack([x_num_scaled, x_cat_arr])
            prediction = model.predict(x_final)[0]
            result = "diabetic" if prediction == 1 else "not diabetic"

    # save record to DB (make sure parameter order matches columns)
    if request.method == "POST":
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
              INSERT INTO predictions (Age, BMI, Glucose, Gender, Smoker, FamilyHistory, Result)
              VALUES (?,?,?,?,?,?,?)
        """, (age, bmi, glucose, gender, smoker, family, result))
        conn.commit()
        conn.close()






    return render_template("index.html", result=result)

@app.route("/records")
def records():
    conn=sqlite3.connect(DB_FILE)
    df=pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    # return a simple template rendering if you have one, otherwise return JSON
    # If you have a `records.html` template, use render_template; otherwise return JSON
    try:
        return render_template("records.html", records=df.to_dict(orient="records"))
    except Exception:
        return jsonify(df.to_dict(orient="records"))
         







def ensure_db():
    if not os.path.exists(DB_FILE):
        init_db()


if __name__ == "__main__":
    ensure_db()
    app.run(debug=True)




