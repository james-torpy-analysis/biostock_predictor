#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#     Title: Identification of biostock crashes following a crash project
#     Author: James Torpy
#     Contact details:james.torpy@gmail.com
#     Date created: 16/10/24
#     Date updated: 
#     Description: Script to download NASDAQ data from Yahoo Finance and detect 
#                  anomalies using isolation forests
#                  To learn how to implement an isolation forest to detect anomalies
#                  in stock prices
#                  https://codime.medium.com/anomaly-detection-in-financial-data-using-isolation-forest-2515b4572bc8
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#########################################################################################
### 1. Load required libraries and functions and set parameters ###
#########################################################################################

import time
from datetime import date

# downloading data
import os
import csv
import urllib.request
import yfinance as yf

# predicting anomalies and crashes
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# import functions
from functions import get_symbols
from functions import download_stock_data
from functions import predict_anomalies
from functions import plot_anomalies
from functions import extract_first_indices
from functions import find_crashes

todays_date = date.today().strftime("%d_%m_%y")

home_dir = "/Users/jamestorpy/Desktop"
project_dir = os.path.join(home_dir, "machine_learning/biostock_prediction")
out_dir = os.path.join(project_dir, "results/past_stock_data", todays_date)

os.makedirs(out_dir, exist_ok=True)

data_window = 30   # number of past days to download
batch_size = 10 # number of symbols to download from in one go

cont_val = 0.01 # estimate of outlier contamination for isolation forest method
perc_drop_req = 30 # percentage stock required to drop before it is identified as a crash
perc_drop_time = 7 # minimum time period stock required to drop before it is identified as a crash


#########################################################################################
### 2. Download and parse all NASDAQ stocks ###
#########################################################################################

# fetch and parse symbols of all NASDAQ stock symbols
nasdaq_symbols = get_symbols()

# download all data in batches
start_time = time.time()
nasdaq_data = download_stock_data(nasdaq_symbols, data_window, batch_size, out_dir)
end_time = time.time()
end_time - start_time
    

#########################################################################################
### 3. Detect anomalies ###
#########################################################################################

predicted_anomalies = {key: predict_anomalies(df, cont_val = 0.01, acolname = 'Anomalies_0.01_contamination') for 
    key, df in nasdaq_data.items()}

results = {key: predict_anomalies(df, cont_val = 0.01, acolname = 'Anomalies_0.01_contamination') for 
    key, df in nasdaq_data.items()}

find_crashes(predicted_anomalies['GPCR'], crash_data['GPCR'])