
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#     Title: Functions for identification biostock crashes
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


def split_stock_data(stock_df):

    '''Splits the stock data output into a df for each symbol.

    Args:
        lst: A list of integers.

    Returns:
        A list of the first indices of concurrent sequences.
    '''
    
    import pandas as pd
    
    # get symbols
    symbols = list(stock_df['Adj Close'].columns)
    
    # get unique level 0 values
    unique_levels = stock_df.columns.get_level_values(0).unique()
    
    # create dictionary with empty dataframes to catch values
    split_data = {}
    for symbol in symbols:
        split_data[symbol] = pd.DataFrame()
    
    # for each symbol, grab each level of data and bind it to the symbol's data frame
    for symbol in symbols:
        for level in unique_levels:
            # select level
            df_level = stock_df[level]
            # add values to dictionary for symbol
            split_data[symbol][level] = df_level[symbol]
            
    return split_data


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


def plot_anomalies(data_df, mcolname, acolname, drop_ranges = None):

    '''Plots anamolies detected from NASDAQ stock data dfs on a time series line plot.

    Args:
        
        data_df: A NASDAQ stock data df outputted from download_stock_data, with a 
                 boolean column identifying anomalies.
        mcolname: The name of the values column to be plotted on the y-axis 
                  (Adj Close, Volume or VolumeClose)
        acolname: The name of the anomaly output column
        drop_ranges: Should any anomalies marked as drops in a boolean'drops' column 
                     within data_df (i.e. marked as True) be highlighted with a red
                     line?

    Returns:
        A time series line plot of stock prices/volumes, with anomalies marked in red.
    '''

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(data_df.index, data_df[mcolname], label=mcolname)

    if drop_ranges is not None:
        ax.plot(data_df.index[data_df['drop'] == True], data_df[mcolname][data_df['drop'] == True], \
            color = 'red', linewidth=2, label='Notable crash')

    ax.scatter(data_df[data_df[acolname] == 1].index, data_df[data_df[acolname] == 1][mcolname], color='red')
    ax.legend([mcolname, acolname])
    
    return(fig)


def find_crashes(anomaly_df, orig_df):

    """Finds crashes from a df of anomalies and reports the pre-crash, start of 
    crash and end of crash prices.

    Args:
        anomaly_df: A dataframe of anomalies for a company outputted from the 
        predict_anomalies function.

    Returns:
        A dataframe with the date and time of the crashes including the pre-crash, 
        start of crash and end of crash prices.
    """

    # fetch array of anomaly close values
    anomaly_ind = np.where(anomaly_df['Anomalies_0.01_contamination'] == 1)[0]
    anomalies_df = anomaly_df.iloc[anomaly_ind,]
    anomaly_closes = anomalies_df['Close'].tolist()

    # fetch array of the values of the datapoints right before the anomalies
    preanomaly_df = anomaly_df.iloc[anomaly_ind-1,]
    preanomaly_closes = preanomaly_df['Close'].tolist()

    # find the indices of the crashes, where the anomaly close value is less than 
    # the preanomaly close value
    crash_logical = [anomaly < preanomaly for anomaly, preanomaly in \
        zip(anomaly_closes, preanomaly_closes)]
    crash_ind = anomaly_ind[crash_logical]

    # check and calculate close differences of crashes (preanomaly close < anomaly 
    # close)
    close_diffs = pd.concat([anomaly_df['Close'][crash_ind-1].reset_index(drop = True), \
        anomaly_df['Close'][crash_ind].reset_index(drop = True)], axis = 1)
    close_diffs.index = anomaly_df['Close'][crash_ind].index

    # determine the indices of anomalies in orig_df
    anomaly_ind = [np.where(orig_df.index == c)[0][0] for c in close_diffs.index]

    # take only the first index of consecutive sequences of indices
    anomaly_ind2 = extract_first_indices(anomaly_ind)

    # subset anomaly_ind and close_diffs
    anomaly_ind = [ind for i, ind in enumerate(anomaly_ind) if i in anomaly_ind2]
    close_diffs = close_diffs.iloc[anomaly_ind2,]

    # insert the preanomaly dates using anomaly_ind - 1
    preanomaly_ind = [ind-1 for ind in anomaly_ind]
    close_diffs.insert(0, 'preanomaly_date', orig_df.index[preanomaly_ind])
    close_diffs.columns = ['preanomaly_date', 'preanomaly_close', 'anomaly_close']
    close_diffs['close_difference'] = close_diffs['anomaly_close'] - close_diffs['preanomaly_close']
    close_diffs['percent_difference_from_preanomaly'] = abs(close_diffs['close_difference']/close_diffs['preanomaly_close']*100).round(1)

    # filter for those with a drop of at least perc_drop_req within perc_drop_time
    # create output table
    potential_drop_list = []

    # iterate through the close_diffs rows
    j=0
    for i, row in close_diffs.iterrows():
        
        # identify index of crash_data with same date as row plus next perc_drop_time rows
        pre_ind = np.where(row['preanomaly_date'] == orig_df.index)[0][0]
        row_inds = range(pre_ind, (pre_ind+perc_drop_time))

        # fetch the dates and closes of these timepoints
        crash_vals = orig_df.iloc[row_inds, ]
        crash_dates = crash_vals.index
        crash_closes = crash_vals['Close']
        
        # calculate percentage changes from preanomaly close
        perc_changes = [((close - row['preanomaly_close'])/row['preanomaly_close'])*100 for close in list(crash_closes)]
        
        # determine which passed threshold
        thresh_passes = [(-1*drop) > perc_drop_req for drop in perc_changes]
        
        # add to output table
        for crash_date, crash_close, perc_change, thresh_pass in zip(crash_dates, crash_closes, perc_changes, thresh_passes):
            changes_to_concat = pd.DataFrame({
                'date': [crash_date],
                'close': [crash_close],
                'percent_change': [perc_change],
                'significant_drop': [thresh_pass]
            })
            if changes_to_concat['percent_change'].iloc[0] == 0:
                j = j+1
            changes_to_concat['anomaly'] = j

            potential_drop_list.append(changes_to_concat)

    potential_drops = pd.concat(potential_drop_list, ignore_index = True)

    # fetch the first and last dates of each true drop
    potential_drops['first_drop'] = potential_drops['significant_drop'] & \
        (~potential_drops['significant_drop'].shift(1, fill_value=False))
    potential_drops['last_drop'] = potential_drops['significant_drop'] & \
        (~potential_drops['significant_drop'].shift(-1, fill_value=False))

    first_drops = potential_drops.iloc[np.where(potential_drops['first_drop'])[0], ][['date', 'close']]
    first_drops.columns = ['start_date', 'start_close']
    first_drops = first_drops.reset_index()

    last_drops = potential_drops.iloc[np.where(potential_drops['last_drop'])[0], ][['date', 'close']]
    last_drops.columns = ['end_date', 'end_close']
    last_drops = last_drops.reset_index()

    # fetch the pre-drop closes - need to get from original anomaly_df, not potential drops!
    pre_drops = potential_drops.iloc[np.where(potential_drops['first_drop'])[0]-1, ][['date', 'close']]
    pre_drops.columns = ['predrop_date', 'predrop_close']
    pre_drops = pre_drops.reset_index()

    drops = pd.concat([
        pre_drops,
        first_drops, 
        last_drops],
        axis=1)
    
    # calculate the initial and overall percentage drops
    drops['percent_initial_drop'] = (drops['start_close'] - drops['predrop_close'])/drops['predrop_close']
    drops['percent_overall_drop'] = (drops['end_close'] - drops['predrop_close'])/drops['predrop_close']

    # filter columns
    drops = drops[['predrop_date', 'predrop_close', 'start_date', 'start_close', 'percent_initial_drop', \
        'end_date', 'end_close', 'percent_overall_drop']]
    
    # add start drop and end drop columns to orig_df
    orig_df['start_drop'] = [dateentry in list(drops['start_date']) for dateentry in orig_df.index]
    orig_df['end_drop'] = [dateentry in list(drops['end_date']) for dateentry in orig_df.index]

    # add drop and crash_no columns to orig_df
    orig_df['drop'] = False
    for i in range(0,drops.shape[0]):
        mask1 = orig_df.index >= drops.iloc[i,]['predrop_date']
        mask2 = orig_df.index <= drops.iloc[i,]['end_date']
        orig_df['drop'][mask1 & mask2] = True

    orig_df['crash_no'] = orig_df['start_drop'].cumsum().where(orig_df['drop'], np.nan)
    
    # plot stocks with start drop marked, and whole drop shaded
    close_drop_plot = plot_anomalies(orig_df, mcolname = 'Adj Close', \
        acolname = 'start_drop', drop_ranges = orig_df['drop'])
    vol_drop_plot = plot_anomalies(orig_df, mcolname = 'Volume', \
        acolname = 'start_drop', drop_ranges = orig_df['drop'])
    volclose_drop_plot = plot_anomalies(orig_df, mcolname = 'VolumeClose', \
        acolname = 'start_drop', drop_ranges = orig_df['drop'])
    return(drops, close_drop_plot, vol_drop_plot, volclose_drop_plot)
