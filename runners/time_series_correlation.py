"""
The purpose of this is to see if the tone of articles that are grouped in some way correlate at all with some economic
metric

stock data is just downloaded as a csv from yahoo finance
tone data comes from this query:
SELECT DATE, V2Tone, V2Organizations FROM `gdelt-bq.gdeltv2.gkg`
WHERE DATE BETWEEN 20201001000000 AND 20201126999999
AND V2Organizations LIKE '%Lloyds%'
LIMIT 9999
"""

import pandas as pd

stock_df = pd.read_csv('data/LLOY.L.csv')
tone_df = pd.read_csv('data/lloyds_Oct01_Oct26_gkg.csv')
tone_df = pd.concat([tone_df[['DATE']], tone_df['V2Tone'].str.split(',',expand=True)], axis=1)
tone_df['DATE'] = [str(x)[:-6] for x in tone_df['DATE']]
for col in tone_df.columns[1:]:
    tone_df[col] = tone_df[col].astype(float)

mean_df = tone_df.groupby(['DATE']).mean()
stock_df['close_diff'] = stock_df['Close']-stock_df['Close'].shift(1)
stock_df['Date'] = [x.replace('-', '') for x in stock_df['Date']]
stock_df = stock_df.set_index(['Date'], drop=True)
master_df = mean_df.join(stock_df[['close_diff']], how='inner')
close_diff_corr = master_df.corr()['close_diff']

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

"""
lets inspect the strongest of the correletions, thats with tone 6
"""

df = master_df.copy()
df['date'] = df.index

scaler = MinMaxScaler(feature_range=(-1,1))
df['close_diff_norm'] = scaler.fit_transform(df[['close_diff']])

scaler = MinMaxScaler(feature_range=(-1,1))
df['6_norm'] = scaler.fit_transform(df[[6]])

temp_df = df[['date', 'close_diff_norm', '6_norm']].melt('date', var_name='cols', value_name='vals')
g = sns.factorplot(x="date", y="vals", hue='cols', data=temp_df)

"""
try another column, col 0 is the second strongest
"""
df = master_df.copy()
df['date'] = df.index

scaler = MinMaxScaler(feature_range=(-1,1))
df['close_diff_norm'] = scaler.fit_transform(df[['close_diff']])

scaler = MinMaxScaler(feature_range=(-1,1))
df['0_norm'] = scaler.fit_transform(df[[0]])

temp_df = df[['date', 'close_diff_norm', '0_norm']].melt('date', var_name='cols', value_name='vals')
g = sns.factorplot(x="date", y="vals", hue='cols', data=temp_df)




