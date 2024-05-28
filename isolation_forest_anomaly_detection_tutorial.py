#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#     Title: Anomaly Detection on Google Stock Data 2014-2022
#     Author: James Torpy
#     Contact details:james.torpy@gmail.com
#     Date created: 24/5/24
#     Date updated: 
#     Description: Anomaly Detection on Google Stock Data 2014-2022
#                  To learn how to implement an isolation forest to detect anomalies in stock prices
#                  https://www.analyticsvidhya.com/blog/2023/02/anomaly-detection-on-google-stock-data-2014-2022/
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

home_dir = "/Users/jamestorpy/Desktop"
project_dir = os.path.join(home_dir, "machine_learning/biostock_prediction")
in_dir = os.path.join(project_dir, "raw_files")



################################################################################
### 1. Load and clean the dataset  ###
################################################################################

# load and check for missing values
orig_data = pd.read_excel(os.path.join(in_dir, 'Google Dataset.xlsx'))
print(orig_data.head())
print(orig_data.isnull().sum())
data = orig_data

# Finding data points that have a 0.0% change from the previous month’s value:
data[data['Change %']==0.0]

# Changing the ‘Month Starting’ column to a date datatype and checking for missing values:
data['Month Starting'] = pd.to_datetime(data['Month Starting'], errors='coerce').dt.date
print(data.isnull().sum())

# Check the missing value rows
missing_dates = np.where(data['Month Starting'].isnull())
data.iloc[missing_dates[0], :]
orig_data.iloc[missing_dates[0], :]

# Replacing the missing values after cross verifying
data.loc[31, 'Month Starting'] = pd.to_datetime('2020-05-01')
data.loc[43, 'Month Starting'] = pd.to_datetime('2019-05-01')
data.loc[55, 'Month Starting'] = pd.to_datetime('2018-05-01')


# Convert to Month Starting to datetime objects
data['Month Starting'] = pd.to_datetime(data['Month Starting'])
# Sort by month
data.sort_values(by='Month Starting', ascending = True, inplace = True)

print(data)


################################################################################
### 2. Exploratory data analysis  ###
################################################################################

plt.figure(figsize=(25,5))
plt.plot(data['Month Starting'],data['Open'], label='Open')
plt.plot(data['Month Starting'],data['Close'], label='Close')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend()
plt.title('Change in the stock price of Google over the years')


# Calculating the daily returns
data['Returns'] = data['Close'].pct_change()

# Calculating the rolling average of the returns
data['Rolling Average'] = data['Returns'].rolling(window=30).mean()

# Creating a line plot using the 'Month Starting' column as the x-axis 
# and the 'Rolling Average' column as the y-axis
plt.figure(figsize=(10,5))

sns.lineplot(x='Month Starting', y='Rolling Average', data=data)

# Scale returns data to have a mean of 0 and a sd of 1
scaler = StandardScaler()
data['Returns'] = scaler.fit_transform(data['Returns'].values.reshape(-1,1))

# Check scaling
data.head()
data['Returns'].mean(skipna=True)
data['Returns'].std(skipna=True)




# Fill NaN values with the mean of the columns
data = data.replace('nan', np.nan)
np.where(data['Returns'].isnull())
data['Returns'] = data['Returns'].fillna(data['Returns'].mean(skipna=True))

np.where(data['Rolling Average'].isnull())
data['Returns'] = data['Rolling Average'].fillna(data['Rolling Average'].mean(skipna=True))
np.where(data['Rolling Average'].isna())
data.loc[105,'Rolling Average'].nan_to_num(data.loc[105,'Rolling Average'], nan = 0)