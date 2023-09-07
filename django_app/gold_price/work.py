import tensorflow as tf 
import numpy as np 
import pandas as pd 
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.firefox import firefox_binary
from selenium.webdriver import Firefox, FirefoxOptions
from datetime import datetime
from tensorflow.keras.models import load_model 
from prophet import Prophet
from numpy.lib.stride_tricks import sliding_window_view
import glob 
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os 
import glob  
from django_app import settings 



def downloading_historical_data():  
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    import time
    import os


    options = Options()
    options.add_argument("--headless=new")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("prefs", {"download.default_directory": r"C:\Users\stadl\MACHINE_LEARNING_PRACTICE\Saját\Stock_Forecasting\django_app\gold_price\new_data"})
    # Check if today is Saturday or Sunday
    today = datetime.today().strftime("%A")
    
    if today == "Sunday":
        now = datetime.now() - timedelta(days=2)
        date = now.strftime("%m/%d/%Y")
    else:
        now = datetime.now() - timedelta(days=1)
        date = now.strftime("%m/%d/%Y")
        
        
    month_earlier = datetime.now() - timedelta(days=31)
    month_earlier = month_earlier.strftime("%m/%d/%Y")
    
    string = "https://www.marketwatch.com/investing/future/gc00/downloaddatapartial?startdate=" + month_earlier + "%2000:00:00&enddate=" + date+ "%2023:59:59&daterange=d30&frequency=p1d&csvdownload=true&downloadpartial=false&newdates=false"

    url = string

    # Set up the Selenium webdriver (you may need to specify the path to your chromedriver executable)
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(5)
    # # Click the "Download to Spreadsheet" button
    # driver.find_element(By.XPATH, "//div[contains(text(), 'Dowload Data - FUTURE/US/XNYM/GC00.csv')]").click()
    # link = driver.find_element(By.XPATH, "//a[@rel='nofollow']")
    # link.click()
    # driver.implicitly_wait(5)

    # Wait for the download to complete (adjust the sleep time as necessary)
    from time import sleep as wait
    wait(10)
    driver.quit()
    # Read the downloaded CSV file into a pandas DataFrame
    # Print the first few rows of the DataFrame
    # Close the webdriver
    # string2 = "\nThe new historical data downloaded"
    now = datetime.now().strftime("%Y%m%d_%H%M")
    

    
    old_file = r"C:\Users\stadl\MACHINE_LEARNING_PRACTICE\Saját\Stock_Forecasting\django_app\gold_price\new_data\downloaddatapartial.csv"
    
    new_file = r"C:\Users\stadl\MACHINE_LEARNING_PRACTICE\Saját\Stock_Forecasting\django_app\gold_price\new_data\Historical_data" + now + ".csv"
    
    os.rename(old_file, new_file)

    
    
def add_new_date():
    """
    Download new historical gold data from 'marketwatch.com' and add the new days that are missing from the Dataframe 
    up to today, so the training data will be big enough to make predictions on"""
    
    csv_path = r"C:\Users\stadl\MACHINE_LEARNING_PRACTICE\Saját\Stock_Forecasting\django_app\gold_price\data_to_predict\Gold_pred.csv"
    #import 'old' gold data and find the last date in the csv 
    old_data = pd.read_csv(csv_path)
    
    # check the last day on old data and get the index value (because I saved the Date column as the index value)
    last_date = old_data.iloc[-1][0]

    
    # Downloading the new gold prices in a csv format
    downloading_historical_data()
    
    # Find the new downloaded file
    my_path  ="C:/Users/stadl/MACHINE_LEARNING_PRACTICE/Saját/Stock_Forecasting/django_app/gold_price/new_data/*.*"
    new = glob.glob(my_path)
    latest_file = max(new, key=os.path.getmtime)
    new_gold_data = pd.read_csv(latest_file)
    new_gold_data["Date"] = pd.to_datetime(new_gold_data['Date'])
    new_data = new_gold_data.rename(columns={"Close": "Price"})
    new_data = new_data.drop(columns=["Open", "High", "Low"])
  
    
    # check if today is a weekend or a weekday 
    #if today is Saturday or Sunday add it to the end to the Dataframe and then fill it with the Friday value
    weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    today = datetime.today().strftime("%A")
    
    today1 = datetime.now().strftime("%Y-%m-%d")
    today1 = pd.to_datetime(today1)
    day0 = pd.DataFrame({"Date": today1, "Close": np.nan}, index=[0])
    if today not in weekday:
        new_gold  = pd.DataFrame([new_data, day0], ignore_index=True)
        new_gold["Date"] = new_gold["Date"].apply(lambda x: datetime.fromisoformat(x).date())
        new_gold.set_index('Date', inplace=True)
        new_gold = new_gold.reindex(pd.date_range(new_gold.index.min(), new_gold.index.max())).sort_index(ascending=True).reset_index().rename(columns={'index': 'Date'})
        new_gold = new_gold.fillna(method="ffill")
        new_gold["Price"]  = new_gold["Price"].apply(lambda x: x.replace(",", ""))
        new_gold["Price"] = new_gold["Price"].astype(float)
        
        
        # Work only with the new gold data that are missing from the old data
        new_gold = new_gold.reset_index()
        num_index = new_gold[new_gold["Date"]== last_date].index.item()
        num_index = int(num_index) +1
        new_gold = new_gold.iloc[num_index:]
        new_d = pd.concat([old_data, new_gold])
        
    else:
        new_data.set_index('Date', inplace=True)
        new_data = new_data.reindex(pd.date_range(new_data.index.min(), new_data.index.max())).sort_index(ascending=True).reset_index().rename(columns={'index': 'Date'})
        new_data = new_data.fillna(method="ffill")
        new_data.Price = new_data["Price"].apply(lambda x: x.replace(",", ""))
        new_data.Price = new_data["Price"].astype(float)
        
        
        # Work only with the new gold data that are missing from the old data
        new_data = new_data.reset_index()
        num_index = new_data[new_data["Date"]== last_date].index.values
        num = int(num_index) + 1
        new_data = new_data.iloc[num:]      
        new_d = pd.concat([old_data, new_data])
        if "index" in new_d.columns:
            new_d = new_d.drop(columns=["index"])
            new_d["Date"] = pd.to_datetime(new_d["Date"])
        else:
            new_d["Date"] = pd.to_datetime(new_d["Date"])
    
    t = os.listdir("C:/Users/stadl/MACHINE_LEARNING_PRACTICE/Saját/Stock_Forecasting/django_app/gold_price/data_to_predict")
    
    if "Gold_pred.csv" in t:
        os.remove("C:/Users/stadl/MACHINE_LEARNING_PRACTICE/Saját/Stock_Forecasting/django_app/gold_price/data_to_predict/Gold_pred.csv")
        new_d.to_csv("C:/Users/stadl/MACHINE_LEARNING_PRACTICE/Saját/Stock_Forecasting/django_app/gold_price/data_to_predict/Gold_pred.csv", index=False)
    else:
        new_d.to_csv("C:/Users/stadl/MACHINE_LEARNING_PRACTICE/Saját/Stock_Forecasting/django_app/gold_price/data_to_predict/Gold_pred.csv", index=False)
    
    
             
        
def make_prediction_with_prophet(days: int, csv_file):
    """Import the newest downloaded historical data and make
    predictions with prophet for 1 and 7 days in advance"""
    pro = Prophet()
    df = pd.read_csv(csv_file)
    df = df.rename(columns={"Date": "ds", "Price": "y"})
    pro.fit(df)
    if days == 1:
        future_gold1 = pro.make_future_dataframe(days)
        gold_pred1 = pro.predict(future_gold1)
        gold1 = gold_pred1[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        pred1 = gold1.iloc[-1]
        g1_dict = {"Time": pred1[0], "Prediction (Price)": pred1[1], "Range_low (Price)": pred1[2], "Range_high (Price)": pred1[3]}
        return g1_dict
    else:
        future_gold = pro.make_future_dataframe(days)
        gold_pred = pro.predict(future_gold)
        gold = gold_pred[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        pred = gold.iloc[-days:]
        time_list = []
        for i in pred.ds.values:
            i = np.datetime_as_string(i, unit="D")
            time_list.append(i)
        g_dict = {"Time": time_list, "Prediction (Price)" : pred.yhat.values, "Range_low (Price)": pred.yhat_lower.values, "Range_high (Price)": pred.yhat_upper.values}
        return g_dict



def get_predictions_model4(model):
    df = pd.read_csv("data_to_predict/Gold_pred.csv")
    df = df.iloc[-3645:]
    df = df.set_index('Date', inplace=True)
    scaler = MinMaxScaler()
    d4 = scaler.fit_transform(df)
    d4 = d4.reshape(1, -1)
    prediction4 = model.predict(d4)
    pred4 = scaler.inverse_transform(prediction4)
    return pred4

def get_predictions_model6(model):
    df = pd.read_csv("gold_price/data_to_predict/Gold_pred.csv")
    df = df.iloc[-3600:]
    df = df.set_index('Date', inplace=True)
    scaler = MinMaxScaler() 
    d6 = scaler.fit_transform(df)
    d6 = d6.reshape(1, -1)
    prediction6 = model.predict(d6)
    pred6 = scaler.inverse_transform(prediction6)
    return pred6 

