
import joblib
import numpy as np
from flask import Flask, request, render_template


app = Flask(__name__)

# Load model artifacts with basic error handling so the app doesn't crash at import time
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    encoder = joblib.load("onehot_encoder.joblib")
except Exception as e:
    # If files are missing or corrupted, set to None and log the error (printed to console)
    model = scaler = encoder = None
    print(f"Error loading model artifacts: {e}")

numeric_cols = ["Age", "BMI", "Glucose"]
categorial_cols = ["Gender", "Smoker", "FamilyHistory"]

@app.route("/", methods=["GET","POST"])
def index():
    result= None
    if request.method  == "POST":
        x_num= np.array([
            float(request.form["Age"]),
            float(request.form["BMI"]),
            float(request.form["Glucose"])

        ]).reshape(1,-1)

        x_cat = np.array([
            request.form["Gender"],
            request.form["Smoker"],
            request.form["FamilyHistory"]
        ]).reshape(1, -1)

        # Ensure model artifacts are available
        if None in (model, scaler, encoder):
            result = "Error: model or preprocessing artifacts not loaded. Check server logs."
        else:
            x_num_scaled = scaler.transform(x_num)
            x_cat_encoded = encoder.transform(x_cat)
            # encoder.transform may return a sparse matrix; convert to array if needed
            try:
                x_cat_arr = x_cat_encoded.toarray() if hasattr(x_cat_encoded, "toarray") else x_cat_encoded
            except Exception:
                x_cat_arr = x_cat_encoded

            x_final = np.hstack([x_num_scaled, x_cat_arr])

            prediction = model.predict(x_final)[0]
            result = "diabetic" if prediction == 1 else "not diabetic"




    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

