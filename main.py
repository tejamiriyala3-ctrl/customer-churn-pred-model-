import joblib 
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI

model=joblib.load('churn_pred.pkl')
scaler = joblib.load("Stands.pkl")
app = FastAPI()

class churn(BaseModel):
    AccountWeeks       : float      
    ContractRenewal    : float
    DataPlan           : float
    DataUsage          : float
    CustServCalls      : float
    DayMins            : float
    DayCalls           : float
    MonthlyCharge      : float
    OverageFee         : float
    RoamMins           : float

@app.get ("/")
def home ():
    return{"message":'your churn prediction api is running '}

@app .post ('/predict')
def predict (data: churn ):
    input_data = np.array([[
        data.AccountWeeks , 
        data.ContractRenewal,
        data.DataPlan,
        data.DataUsage,
        data.CustServCalls,
        data.DayMins,
        data.DayCalls,
        data.MonthlyCharge,
        data.OverageFee,
        data.RoamMins

    ]])

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)

    pred = int(prediction[0])
    proba = float(model.predict_proba(scaled)[0][1])
    return {
        "churn": pred,
        "churn_probability": round(proba, 4),
        "message": "customer likely to churn" if pred == 1 else "customer likely to stay",
    }