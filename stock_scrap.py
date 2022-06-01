# -*- coding: utf-8 -*-
"""

"""

import os
import datetime
import pandas as pd
import yfinance as yf
from functools import reduce
import mplfinance as mpl


# define directories:
home_dir = '/Volumes/GoogleDrive/My Drive/work/'
project_dir = home_dir + 'biostock_predictor/'
ref_dir = project_dir + 'refs/'
plot_dir = project_dir + 'plots/'

os.makedirs(plot_dir, exist_ok=True)


########################################################
### 1. Identify crashes ###
########################################################

# load list of NASDAQ companies:
with open(ref_dir + 'nasdaq_listings.csv') as nasinfo:
    nassym = [row.split(',')[0] for row in nasinfo]

# remove colname:
nassym.pop(0)

# define time window:
now_date = datetime.datetime.now()
week_before = now_date - datetime.timedelta(days = 7)

start_date = (week_before - datetime.timedelta(days=2)).date()
end_date = (now_date - datetime.timedelta(days=2)).date()

start_str = start_date.strftime('%Y-%m-%d')
end_str = end_date.strftime('%Y-%m-%d')

### choose custom date to detect biogen ###
# start_date = '2019-03-14'
# end_date = '2019-03-22'
# nassym = nassym[0:5] + ['BIIB']
###

# download stock data:
#nassym = nassym[800:1000]
nasprice = [list(yf.download(sym,start_date,end_date)['Adj Close'].round(2)) for 
          sym in nassym]

# remove blank values for values and stock codes:
nonempty = [i for i,x in enumerate(nasprice) if x]
nasprice = [nasprice[ind] for ind in nonempty]
naskeep = [nassym[ind] for ind in nonempty]

# fetch prices of past week for each symbol - ### put into one command ###:
all_attr = ['Open', 'High', 'Low', 'Close', 'Adj Close']
plot_attr = ['Open', 'High', 'Low', 'Close']

plot_prices = []
adj_closes = []
for i, sym in enumerate(naskeep):
    print(sym + ' (' + str(i) + '/' + str(len(naskeep)) + ')')
    temp_df = yf.download(sym,start_date,end_date)[all_attr].round(2)
    plot_prices.append(temp_df[plot_attr])
    adj_closes.append(temp_df[['Adj Close']])
    
# select adj close only and merge together above dfs by date column:
close_df = reduce(lambda x, y: pd.merge(x, y, on = 'Date', how = 'outer'), adj_closes)
close_df.columns = naskeep

# remove rows not within correct date range:
keep_rows = [i for i, d in enumerate(close_df.index.to_pydatetime()) if 
             d.date() >= start_date and d.date() <= end_date]
close_df = close_df.iloc[keep_rows]

# remove columns with only one value:
close_df[close_df.columns[close_df.isna().sum() < len(close_df)]]

# keep stocks if last price of week has dropped 25% from first price of week:
keep = list(close_df.apply(lambda x: x[len(x)-1] < 0.75*x[0]))
close_df = close_df.iloc[:,keep]
plot_df = [plot_prices[i] for i,x in enumerate(keep) if x]

crash_sym = [col for col in close_df.columns]
print(crash_sym)


########################################################
### 2. Plot ###
########################################################

# plot last week of prices on candlestick plot:    
[mpl.plot(pl, type="candle",title = crash_sym[i], mav=2,
    style="yahoo", savefig = plot_dir + crash_sym[i] + 
    '_prev_week.png') for i,pl in enumerate(plot_df)]

### choose custom date to detect biogen ###
#today = datetime.datetime.strptime('2019-03-22', '%Y-%m-%d')
###

# plot last month, 6 months and year on candlestick plot:
today = datetime.date.today()

month_start = today - datetime.timedelta(days=31)
month_start = month_start.strftime('%Y-%m-%d')
month_prices = [yf.download(sym,month_start,end_date)[plot_attr].round(2) for 
                sym in crash_sym]
[mpl.plot(pl, type="candle",title = crash_sym[i], mav=2, 
          style="yahoo", savefig = plot_dir + crash_sym[i] + 
          '_prev_month.png') for i,pl in 
     enumerate(month_prices)]

half_start = today - datetime.timedelta(days=183)
half_start = half_start.strftime('%Y-%m-%d')
half_prices = [yf.download(sym,half_start,end_date)[plot_attr].round(2) for 
                sym in crash_sym]
[mpl.plot(pl, type="candle",title = crash_sym[i], mav=2, 
          style="yahoo", savefig = plot_dir + crash_sym[i] + 
          '_prev_half_year.png') for i,pl in 
     enumerate(half_prices)]

year_start = today - datetime.timedelta(days=365)
year_start = year_start.strftime('%Y-%m-%d')
year_prices = [yf.download(sym,year_start,end_date)[plot_attr].round(2) for 
                sym in crash_sym]
[mpl.plot(pl, type="candle",title = crash_sym[i], mav=2, 
          style="yahoo", savefig = plot_dir + crash_sym[i] + 
          '_prev_year.png') for i,pl in 
     enumerate(year_prices)]


