
# Functions for prediction of biostock recovery following a crash project

#########################################################################################
### Functions to download data ###
#########################################################################################

# function to download NASDAQ symbols and parse into a list
def get_symbols():
    
    import urllib.request
    
    # download symbols
    url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
    response = urllib.request.urlopen(url)
    data = response.read().decode("utf-8")
    
    # parse symbols
    symbols = [line.split("|")[0] for line in data.strip().splitlines()[1:]]
    return(symbols)

# function to split the stock data output into a df for each symbol
def split_stock_data(stock_df):
    
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

# function to download nasdaq stock data in batches
def download_stock_data(symbol_list, data_window, batch_size, out_dir):
    
    import yfinance as yf
    import os
    import csv
    
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

#########################################################################################
### Functions to detect anomalies ###
#########################################################################################

# function to predict anomalies
def predict_anomalies(data_df, cont_val, acolname):

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
def plot_anomalies(data_df, mcolname, acolname):
    plt.figure(figsize=(13,5))
    plt.plot(data_df.index, data_df[mcolname], label=mcolname)
    plt.scatter(data_df[data_df[acolname] == 1].index, data_df[data_df[acolname] == 1][mcolname], color='red')
    plt.legend([mcolname, acolname])
    plt.show()

# function to plot anomalies on plots of different metrics 
def plot_anomaly_sets(data_df):
    plot_anomalies(data_df, mcolname = "VolumeClose", acolname = 'Anomalies_0.01_contamination')
    plot_anomalies(data_df, mcolname = "Adj Close", acolname = 'Anomalies_0.01_contamination')
    plot_anomalies(data_df, mcolname = "Volume", acolname = 'Anomalies_0.01_contamination')