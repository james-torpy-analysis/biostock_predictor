#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#     Title: Anomaly Detection in Financial Data Using Isolation Forest
#     Author: James Torpy
#     Contact details:james.torpy@gmail.com
#     Date created: 24/5/24
#     Date updated: 
#     Description: Anomaly Detection in Financial Data Using Isolation Forest
#                  To learn how to implement an isolation forest to detect anomalies in stock prices
#                  https://codime.medium.com/anomaly-detection-in-financial-data-using-isolation-forest-2515b4572bc8
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import numpy as np
import pandas as pd
import datetime
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

home_dir = '/Users/jamestorpy/Desktop'
project_dir = os.path.join(home_dir, 'machine_learning/biostock_prediction')
in_dir = os.path.join(project_dir, 'results')

# set seed for reproducibility
seed = 54

# define contamination value
cont_val = 0.01

# define percentage drop required
perc_drop_req = 30

# create functions
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

def plot_anomalies(data_df, mcolname, acolname):
    plt.figure(figsize=(13,5))
    plt.plot(data_df.index, data_df[mcolname], label=mcolname)
    plt.scatter(data_df[data_df[acolname] == 1].index, data_df[data_df[acolname] == 1][mcolname], color='red')
    plt.legend([mcolname, acolname])
    plt.show()
    
def plot_anomaly_sets(data_df):
    plot_anomalies(data_df, mcolname = "VolumeClose", acolname = 'Anomalies_0.01_contamination')
    plot_anomalies(data_df, mcolname = "Adj Close", acolname = 'Anomalies_0.01_contamination')
    plot_anomalies(data_df, mcolname = "Volume", acolname = 'Anomalies_0.01_contamination')


################################################################################
### 1. Load and clean the dataset  ###
################################################################################

with open(os.path.join(in_dir, 'test_crash_data.pickle'), 'rb') as f:
    crash_data = pickle.load(f)


################################################################################
### 2. Detect anomalies  ###
################################################################################

# train model

'''
n_estimators: the number of estimators, i.e. decision trees, in the ensemble,
max_samples: the number of sub-samples to take for tree training (auto ensures at least 256 sub-samples),
contamination: judgement about the proportion of outliers in data,
random_state: ensures result reproducibility.
'''

# compute anomaly scores, estimating 1% outliers
predicted_anomalies = {key: predict_anomalies(df, cont_val = 0.01, acolname = 'Anomalies_0.01_contamination') for 
    key, df in crash_data.items()}

# plot anomalies
[plot_anomaly_sets(df) for df in predicted_anomalies.values()]


################################################################################
### 3. Filter for crashes  ###
################################################################################

# fetch array of anomaly close values
df = predicted_anomalies['RAPT']
anomaly_ind = np.where(df['Anomalies_0.01_contamination'] == 1)[0]
anomaly_df = df.iloc[anomaly_ind,]
anomaly_closes = anomaly_df['Close'].tolist()

# fetch array of the values of the datapoints right before the anomalies
preanomaly_df = df.iloc[anomaly_ind-1,]
preanomaly_closes = preanomaly_df['Close'].tolist()

# find the indices of the crashes, where the anomaly close value is less than the preanomaly close value
crash_logical = [anomaly < preanomaly for anomaly, preanomaly in zip(anomaly_closes, preanomaly_closes)]
crash_ind = anomaly_ind[crash_logical]

# check and caluclate close differences of crashes (preanomaly close < anomaly close)
close_diffs = pd.concat([df['Close'][crash_ind-1].reset_index(drop = True), df['Close'][crash_ind].reset_index(drop = True)], axis = 1)
close_diffs.columns = ['preanomaly_close', 'anomaly_close']
close_diffs.index = df['Close'][crash_ind].index
close_diffs['close_difference'] = close_diffs['anomaly_close'] - close_diffs['preanomaly_close']
close_diffs['percent_difference'] = abs(close_diffs['close_difference']/close_diffs['preanomaly_close']*100).round(1)

# filter for those with a drop of at least 30% - need to adjust this so it's 30% over a time period of n days!
close_diffs[close_diffs['percent_difference'] >= perc_drop_req]
