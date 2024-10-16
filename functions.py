
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#     Title: Functions for prediction of biostock recovery following a crash project
#     Author: James Torpy
#     Contact details:james.torpy@gmail.com
#     Date created: 16/10/24
#     Date updated: 
#     Description: Functions to download NASDAQ data from Yahoo Finance and detect 
#                  anomalies using isolation forests
#                  To learn how to implement an isolation forest to detect anomalies
#                  in stock prices
#                  https://codime.medium.com/anomaly-detection-in-financial-data-using-isolation-forest-2515b4572bc8
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#########################################################################################
### Load required libraries ###
#########################################################################################

# downloading data
import os
import csv
import urllib.request
import yfinance as yf

# predicting anomalies and crashes
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


#########################################################################################
### Functions to download data ###
#########################################################################################

def get_symbols():

    '''Downloads NASDAQ symbols and parses into a list.

    Returns:
        A list of NASDAQ symbols.
    '''
    
    # download symbols
    url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    response = urllib.request.urlopen(url)
    data = response.read().decode("utf-8")
    
    # parse symbols
    symbols = [line.split("|")[0] for line in data.strip().splitlines()[1:]]
    return(symbols)


# function to 
def download_stock_data(symbol_list, data_window, batch_size, out_dir):
    
    '''Downloads NASDAQ stock data in batches.

    Args:
        symbol_list: A list of NASDAQ symbols to download data for.
        data_window: The number of past days to download stock data for, starting from today.
        batch_size: The number of symbols to download per batch, to avoid blocking from Yahoo.
        out_dir: The output directory to store the data in.

    Returns:
        A dictionary of dfs of stock data, including datetime, adj. close, volume. Each key-value
        pair represents data for a different NASDAQ symbol.
    '''
    
    # create empty dictionary to catch values
    out_dict = {}
    
    # delete any existing download progress csvs
    dprogress_file = f'{out_dir}/download_progress.csv'
    try:
        os.remove(dprogress_file)
        print(f"File '{dprogress_file}' deleted successfully, creating a new one now...")
    except FileNotFoundError:
        print(f"No prior '{dprogress_file}' file found, creating one now...")
            
    # download stock data in batches
    for i in range(0, len(symbol_list), batch_size):
        print('downloading stocks ' + str(i) + '-' + str(i+batch_size))
        batch_symbols = symbol_list[i:i+batch_size]
        out_data = yf.download(batch_symbols, period = str(data_window) + 'd')
 
        # update progress to csv
        with open(dprogress_file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([f'batch_{str(i)}_{str(i+batch_size)} complete'])
            
        # split data by symbol
        out_dict[f'batch_{str(i)}_{str(i+batch_size)}'] = split_stock_data(out_data)
       
    # unnest dictionary, removing grouping by batch
    final_dict = {subkey: value for value in out_dict.values() for subkey, value in value.items()}
    
    return final_dict

# def split_stock_data(stock_df):

#     '''Splits the stock data output into a df for each symbol.

#     Args:
#         lst: A list of integers.

#     Returns:
#         A list of the first indices of concurrent sequences.
#     '''
    
#     import pandas as pd
    
#     # get symbols
#     symbols = list(stock_df['Adj Close'].columns)
    
#     # get unique level 0 values
#     unique_levels = stock_df.columns.get_level_values(0).unique()
    
#     # create dictionary with empty dataframes to catch values
#     split_data = {}
#     for symbol in symbols:
#         split_data[symbol] = pd.DataFrame()
    
#     # for each symbol, grab each level of data and bind it to the symbol's data frame
#     for symbol in symbols:
#         for level in unique_levels:
#             # select level
#             df_level = stock_df[level]
#             # add values to dictionary for symbol
#             split_data[symbol][level] = df_level[symbol]
            
#     return split_data


#########################################################################################
### Functions to detect anomalies ###
#########################################################################################

# function to predict anomalies
def predict_anomalies(data_df, cont_val, acolname):

    '''Predicts anamolies in NASDAQ stock data dfs.

    Args:
        
        data_df: A NASDAQ stock data df outputted from download_stock_data.
        cont_val: A parameter of the IsolationForest function which controls the detection
                  sensitivity. Input value is an estimation of the amount of contamination 
                  of the data set, i.e. the proportion of outliers.
        acolname: The name of the anomaly output column

    Returns:
        The input df data_df with an additional boolean column indicating detection (True) 
        or no detection (False) of anomalies.
    '''

    # Create VolumeClose feature
    data_df['VolumeClose'] = data_df['Adj Close'] * data_df['Volume']
    
    # Train isolation forest model
    if_model = IsolationForest(contamination=cont_val)
    if_model.fit(data_df[['VolumeClose']])
    
    # Predict anomalies
    data_df[acolname] = if_model.predict(data_df[['VolumeClose']])
    data_df[acolname] = data_df[acolname].map({1: 0, -1: 1})
    
    return data_df


# function to plot anomalies on a time series line plot
def plot_anomalies(data_df, mcolname, acolname, drop_ranges = None):

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(data_df.index, data_df[mcolname], label=mcolname)

    if drop_ranges is not None:
        ax.plot(data_df.index[data_df['drop'] == True], data_df[mcolname][data_df['drop'] == True], \
            color = 'red', linewidth=2, label='Notable crash')

    ax.scatter(data_df[data_df[acolname] == 1].index, data_df[data_df[acolname] == 1][mcolname], color='red')
    ax.legend([mcolname, acolname])
    
    return(fig)

def extract_first_indices(lst):
    '''Extracts the first index of concurrent sequences of indices from a list.

    Args:
        lst: A list of integers.

    Returns:
        A list of the first indices of concurrent sequences.
    '''

    first_indices = [0]
  
    for i, val in enumerate(lst):
        if i != 0:
            if lst[i] != lst[i - 1] + 1:  # Check if the current number is not consecutive
                first_indices.append(i)
    return first_indices