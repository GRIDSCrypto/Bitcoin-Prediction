import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

# get market info for bitcoin from the start of 2016 to the current day
bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20160428&end="+time.strftime("%Y%m%d"))[0]
# convert the date string to the correct date format
bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
# when Volume is equal to '-' convert it to 0
# bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0
# convert to int
bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')
# look at the first few rows
bitcoin_market_info.head()

market_info = bitcoin_market_info

kwargs = {
  'day_diff': lambda x: (x['Close']-x['Open'])/x['Open'],
  'close_off_high': lambda x: 2*(x['High']- x['Close'])/(x['High']-x['Low'])-1,
  'volatility': lambda x: (x['High']- x['Low'])/(x['Open'])
}

market_info = market_info.assign(**kwargs)

model_data = market_info[['Date']+[metric for metric in ['Close','Volume','close_off_high','volatility']]]
model_data = model_data.sort_values(by='Date')

model_data.head()

split_date = '2017-06-01'

training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]

training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)

window_len = 10
norm_cols = [metric for metric in ['Close','Volume']]

LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
  temp_set = training_set[i:(i+window_len)].copy()
  for col in norm_cols:
    temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
  LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1


LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
  temp_set = test_set[i:(i+window_len)].copy()
  for col in norm_cols:
    temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
  LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['Close'][window_len:].values/test_set['Close'][:-window_len].values)-1


LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

# random seed for reproducibility
np.random.seed(202)
# initialise model architecture
btc_model = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
# model output is next price normalised to 10th previous closing price
LSTM_training_outputs = (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1
# train model on data
# note: eth_history contains information on the training error per epoch
btc_history = btc_model.fit(LSTM_training_inputs, LSTM_training_outputs, 
                            epochs=50, batch_size=1, verbose=2, shuffle=True)

import pdb; pdb.set_trace()


for rand_seed in range(775,800):
  print(rand_seed)
  np.random.seed(rand_seed)
  temp_model = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
  temp_model.fit(LSTM_training_inputs,
               (training_set['Close'][window_len:].values/training_set['Close'][:-window_len].values)-1,
               epochs=50, batch_size=1, verbose=0, shuffle=True)
  temp_model.save('bt_model_randseed_%d.h5'%rand_seed)


bt_preds = []
for rand_seed in range(775,800):
  temp_model = load_model('bt_model_randseed_%d.h5'%rand_seed)
  bt_preds.append(np.mean(abs(np.transpose(temp_model.predict(LSTM_test_inputs))-
                (test_set['Close'].values[window_len:]/test_set['Close'].values[:-window_len]-1))))

bt_random_walk_preds = []
for rand_seed in range(775,800):
    np.random.seed(rand_seed)
    bt_random_walk_preds.append(
      np.mean(np.abs((np.random.normal(bt_r_walk_mean, bt_r_walk_sd, len(test_set)-window_len)+1)-
                       np.array(test_set['Close'][window_len:])/np.array(test_set['Close'][:-window_len]))))


fig, (ax1) = plt.subplots(1,1)
ax1.boxplot([bt_preds, bt_random_walk_preds],widths=0.75)
ax1.set_ylim([0, 0.2])
ax1.set_xticklabels(['LSTM', 'Random Walk'])
ax1.set_title('Bitcoin Test Set (25 runs)')
ax1.set_ylabel('Mean Absolute Error (MAE)',fontsize=12)
plt.show()

