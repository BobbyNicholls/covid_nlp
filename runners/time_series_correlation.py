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
