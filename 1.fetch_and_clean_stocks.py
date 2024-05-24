
# Phase 1 of prediction of biostock recovery following a crash project, step 1.
# Fetches and cleans NASDAQ stockmarket data for crash detection


import yfinance as yf
import pandas as pd
import datetime


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


company_codes = ['RAPT', 'GPCR', 'KOD']
crash_strings = {
    'RAPT': '2024-02-24',
    'GPCR': '2023-12-18',
    'KOD': '2024-03-29'
}

crash_dates = {key: datetime.datetime.strptime(date_str, '%Y-%m-%d') for key, date_str in crash_strings.items()}




date_ranges = [[]
    for symbol, date in date_ranges.items()]

nasdaq_data = [yf.download(symbol, start=start_date, end=end_date, interval="1h")
    for symbol, (start_date, end_date) in date_ranges.items()]