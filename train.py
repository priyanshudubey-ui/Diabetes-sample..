
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib

df=pd.read_csv("data/diabetes_simple.csv")


X=df.drop(columns=["Diabetes"])
y=df["Diabetes"]

numeric_cols= ["Age","BMI", "Glucose"]
categorical_cols=["Gender","Smoker","FamilyHistory"]

ohe=OneHotEncoder(handle_unknown="ignore" , sparse_output=False)
x_cat=ohe.fit_transform(X[categorical_cols])

scaler=StandardScaler()
x_num=scaler.fit_transform(X[numeric_cols])

x_final=np.hstack([x_num, x_cat])

x_train, x_test, y_train, y_test= train_test_split(
    x_final, y, test_size=0.2, stratify=y, random_state=42

)

model=LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Repost:\n", classification_report(y_test,y_pred))

joblib.dump(model, "model.joblib")
joblib.dump(ohe,"onehot_encoder.joblib")
joblib.dump(scaler,"scaler.joblib")