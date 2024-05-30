
# Phase 1 of prediction of biostock recovery following a crash project, step 1.
# Fetches and cleans NASDAQ stockmarket data for crash detection

import os
import yfinance as yf
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import pickle
from sklearn.ensemble import IsolationForest

crash_window = 12    # number of months around crash date to grab

home_dir = "/Users/jamestorpy/Desktop"
project_dir = os.path.join(home_dir, "machine_learning/biostock_prediction")
out_dir = os.path.join(project_dir, "results")

os.makedirs(out_dir, exist_ok=True)


#########################################################################################
### 1. Download data ###
#########################################################################################

# Downloading the following data:

# a) Massive crash - Rapt Therapeutics (RAPT) – crashed 74% on 20/2/24 as two clinical 
# studies put on hold by FDA – has not recovered

# b) Medium crash – Structure Therapeutics (GPCR) – crashed 43% on 18/12/23 after 
# publishing some clinical trial data the market disliked – half recovery within month, 
# then steady decline for next 5 months

# c) Small crash – Kodiak Sciences (KOD) crashed 33% 27/3 – 1/4/24

# define companies codes and dates of crash
company_codes = ['RAPT', 'GPCR', 'KOD']
crash_strings = {
    'RAPT': '2024-02-24',
    'GPCR': '2023-12-18',
    'KOD': '2024-03-29'
}

# convert to datetime
crash_dates = {key: datetime.datetime.strptime(date_str, '%Y-%m-%d') for 
    key, date_str in crash_strings.items()}

# define window around crash dates
crash_dates = {key: datetime.datetime.strptime(date_str, '%Y-%m-%d') for 
    key, date_str in crash_strings.items()}

crash_windows = {key: [date - relativedelta(months = crash_window/2), 
    date + relativedelta(months = crash_window/2)] for 
    key, date in crash_dates.items()}


crash_data = {key: yf.download(key, start=start_date, 
    end=end_date, interval="1h") for key, (start_date, end_date) in 
    crash_windows.items()}

# save as pickle file
with open(os.path.join(out_dir, 'test_crash_data.pickle'), 'wb') as file:
    pickle.dump(crash_data, file, protocol=pickle.HIGHEST_PROTOCOL)