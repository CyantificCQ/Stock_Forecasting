from django.shortcuts import render 
from tensorflow.keras.models import load_model
from gold_price.work import make_prediction_with_prophet, add_new_date
from prophet import Prophet
import csv
import pandas as pd
import os



csv_path = r"C:\Users\stadl\MACHINE_LEARNING_PRACTICE\Saját\Stock_Forecasting\django_app\gold_price\data_to_predict\Gold_pred.csv"

model4 = load_model(r"./saved_models/Gold_model4.keras")
model6 = load_model(r"./saved_models/Gold_model6.keras")



def prediction(request):
    return render(request, "index.html")


def formInfo(request):
    
    days = request.GET["days_number"]
    days = int(days)
    add_new_date()
    pred = make_prediction_with_prophet(days=days, csv_file=csv_path)
    return render(request, "result.html", {"days": days, "result": pred})

    # pred4 = get_predictions_model4(model=model4)
    # prophet_pred = make_prediction_with_prophet(period=days, csv_file=csv_path)
    # return render(request, "result.html", {"days": str(days)}, {"result": prophet_pred})
   
    