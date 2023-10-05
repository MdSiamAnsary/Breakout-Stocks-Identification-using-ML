# Importing necessary packages
import talib
import yfinance as yf
import numpy as np
import pandas as pd
from collections import Counter
import imblearn.over_sampling
from imblearn.over_sampling import RandomOverSampler

# If CSV file exists, delete it before starting the work
import os
if os.path.exists("data.csv"):
  os.remove("data.csv")
#-------------------------------------------------------

# Fetching stock data using yfinance
data = yf.download("SPY", start="2000-01-01", end="2022-05-20")
print(data.shape)

# Dropping unnecessary columns
data = data.drop(['Adj Close', 'Volume'], axis = 1)
# print(data)

# Introducing additional columns for keeping the CLOSE values from last 15 days 
data['L01'] = 0
data['L02'] = 0
data['L03'] = 0
data['L04'] = 0
data['L05'] = 0
data['L06'] = 0
data['L07'] = 0
data['L08'] = 0
data['L09'] = 0
data['L10'] = 0
data['L11'] = 0
data['L12'] = 0
data['L13'] = 0
data['L14'] = 0
data['L15'] = 0

# Column for the target column 
data['IsBreaking'] = 0

# Converting the dataframe to 2d array
arr = data.to_numpy()

'''
    We have added columns to keep the CLOSE values from last 15 days.
    Hence, we have to iterate to bring those values.
'''
for i in range(15,len(arr)):
    arr[i][4] = arr[i-1][3]
    arr[i][5] = arr[i-2][3]
    arr[i][6] = arr[i-3][3]
    arr[i][7] = arr[i-4][3]
    arr[i][8] = arr[i-5][3]
    arr[i][9] = arr[i-6][3]
    arr[i][10] = arr[i-7][3]
    arr[i][11] = arr[i-8][3]
    arr[i][12] = arr[i-9][3]
    arr[i][13] = arr[i-10][3]
    arr[i][14] = arr[i-11][3]
    arr[i][15] = arr[i-12][3]
    arr[i][16] = arr[i-13][3]
    arr[i][17] = arr[i-14][3]
    arr[i][18] = arr[i-15][3]

# Convert array to dataframe
df = pd.DataFrame(arr, columns=['Open', 'High', 'Low', 'Close',
                                'L01', 'L02', 'L03', 'L04',
                                'L05', 'L06', 'L07', 'L08',
                                'L09', 'L10', 'L11', 'L12',
                                'L13', 'L14', 'L15',
                                'IsBreaking'])

'''
    Dropping first 15 rows. Because, we have 15 days of data from 16th day. 
'''
df = df.drop(labels=range(0,15), axis=0)

'''
    For finding the max and min prices of last 15 days,
    having a temp dataframe where stock prices of last 15 days
    will remain and then, max and min price will be found out using pasdas series.
    then, the series will be saved in dataframes.
'''
temp_df = df.drop(['Open', 'High', 'Low', 'Close', 'IsBreaking'], axis = 1)


# Finding out the max and min values 
maxValuesObj = temp_df.max(axis=1)
minValuesObj = temp_df.min(axis=1)

# Dataframes for max and min values 
mxdf = maxValuesObj.to_frame(name='max')
mndf = minValuesObj.to_frame(name='min')

# Concat the two dataframes of max and min
mxmndf = pd.concat([mxdf, mndf], axis=1)

# Concat the max-min dataframe with df dataframe that has all the columns 
df = pd.concat([df, mxmndf], axis=1)

'''
    At first, we check if consolidation occured. Meaning, if the stock traded maintaining a very
    narrow range in the last 15 days.

    If min close value from last 15 days is greater than 98% of the max close from last 15 days,
    we can say consolidation occured.

    If consolidation occured,
        then, if the close price is greater than max close from last 15 days or
        close price is less than min close from last 15 days,
        we identify the stock as breakout candidate. 
'''
for i in range(len(df)):
    if (df.iloc[i, 21] > df.iloc[i,20] * 0.98):
        if (df.iloc[i, 3] > df.iloc[i,20]):
            df.iloc[i, 19] = 1
        elif (df.iloc[i, 3] < df.iloc[i,21]):
            df.iloc[i, 19] = 1

# We drop the columns with close values from last 15 days
# as they are no longer needed
df = df.drop(['L01', 'L02', 'L03', 'L04',
              'L05', 'L06', 'L07', 'L08',
              'L09', 'L10', 'L11', 'L12',
              'L13', 'L14', 'L15'], axis=1)

# We check how many breakout stocks and how many not breakout stocks
Breaking_Stocks = df[df['IsBreaking'] != 0]
Not_Breaking_Stocks = df[df['IsBreaking'] == 0]

arr = df.to_numpy()
res = pd.DataFrame(arr)

res[4].value_counts()

# Indices of both kinds of stocks 
not_breaking_indices= res[res[4]==0].index
breaking_indices= res[res[4]==1].index

# Undersampling
random_not_breaking_indices = np.random.choice(not_breaking_indices, len(Breaking_Stocks), replace=False)

indices = np.concatenate([random_not_breaking_indices, breaking_indices])
sample = res.loc[indices]
print(sample[4].value_counts())

arr = sample.to_numpy()

# Convert the numpy array to dataframe 
sample = pd.DataFrame(arr, columns = ['Open', 'High', 'Low', 'Close', 'IsBreaking', 'max', 'min'])

print(sample.shape)

# save the dataframe to a csv file 
sample.to_csv("data.csv")
sample.to_excel("output.xlsx")


