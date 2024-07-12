
# Phase 1 of prediction of biostock recovery following a crash project
# Fetches and cleans NASDAQ stockmarket data for crash detection

import os
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import time
from datetime import date
import pickle

# import functions
from functions import get_symbols
from functions import split_stock_data
from functions import download_stock_data

todays_date = date.today().strftime("%d_%m_%y")

home_dir = "/Users/jamestorpy/Desktop"
project_dir = os.path.join(home_dir, "machine_learning/biostock_prediction")
out_dir = os.path.join(project_dir, "results/past_stock_data", todays_date)

os.makedirs(out_dir, exist_ok=True)

data_window = 30   # number of past days to download
batch_size = 10 # number of symbols to download from in one go


#########################################################################################
### 1. Download and parse all NASDAQ stocks ###
#########################################################################################

# fetch and parse symbols of all NASDAQ stock symbols
nasdaq_symbols = get_symbols()

# download all data in batches
start_time = time.time()
nasdaq_data = download_stock_data(nasdaq_symbols, data_window, batch_size, out_dir)
end_time = time.time()
end_time - start_time

# save as pickle file - need to fix this!
pickle_file = os.path.join(out_dir, f'nasdaq_data_{todays_date}.pickle')
with open(pickle_file, 'wb') as file:
    pickle.dump(nasdaq_data, file, protocol=pickle.HIGHEST_PROTOCOL)
    

    
#########################################################################################
### 2. Detect anomalies ###
#########################################################################################

