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
