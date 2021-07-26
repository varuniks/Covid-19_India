import pandas as pd
import numpy as np
from utils import *
from sklearn.preprocessing import MinMaxScaler
from ML_model import fit_model, make_forecasts, eval_forecasts
from plot import plot_state_prediction
import sys


def process_data(state):

    X = pd.read_csv('weekly_india_case_data.csv')
    states_list = X['state'].tolist()
    # process all state data, already present from 4/26/2020 to 07/4/2021, total of 63 weeks
    if state == 'all':
        print("processing for all states")
        X.drop(['state'], axis=1, inplace=True)
    elif state in states_list:
        print(f"processing for state: {state}") 
    else:
        print("state not present")
        sys.exit()
    return X.values, states_list

in_win = 3
out_win = 4
batch_size = 1
n_epochs = 1000
state = 'all'
raw_data, states_list = process_data(state)
raw_data = np.transpose(raw_data)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_values = scaler.fit_transform(raw_data)

X_tr, Y_tr, X_t, Y_t = convert_to_supervised(scaled_values, in_win, out_win, 0.4)

model = fit_model(X_tr, Y_tr, in_win, out_win, n_epochs, batch_size, True, 'ED')

# test prediction
test_prediction = make_forecasts(model, batch_size, X_t, in_win, out_win)
rmse, mape, mae = eval_forecasts(Y_t, test_prediction, in_win, out_win)
print(f"TESTING : rmse:{rmse}, mape:{mape}, mae:{mae}")
   

# train prediction
train_prediction = make_forecasts(model, batch_size, X_tr, in_win, out_win)
rmse, mape, mae = eval_forecasts(Y_tr, train_prediction, in_win, out_win)
print(f"TRAINING : rmse:{rmse}, mape:{mape}, mae:{mae}")

print(train_prediction.shape)
print(test_prediction.shape)
print(Y_tr.shape)
print(Y_t.shape)
    
# rescale

# test prediction
test_prediction = make_forecasts(model, batch_size, X_t, in_win, out_win)
for i in range(len(test_prediction)):
    test_prediction[i,:,:] = np.rint(scaler.inverse_transform(test_prediction[i,:,:]))
    Y_t[i,:,:] = np.rint(scaler.inverse_transform(Y_t[i,:,:]))
rmse, mape, mae = eval_forecasts(Y_t, test_prediction, in_win, out_win)
print(f"TESTING after rescaling: rmse:{rmse}, mape:{mape}, mae:{mae}")
   

# train prediction
train_prediction = make_forecasts(model, batch_size, X_tr, in_win, out_win)
for i in range(len(train_prediction)):
    train_prediction[i,:,:] = np.rint(scaler.inverse_transform(train_prediction[i,:,:]))
    Y_tr[i,:,:] = np.rint(scaler.inverse_transform(Y_tr[i,:,:]))
rmse, mape, mae = eval_forecasts(Y_tr, train_prediction, in_win, out_win)
print(f"TRAINING after rescaling: rmse:{rmse}, mape:{mape}, mae:{mae}")

plot_state_prediction(states_list, train_prediction, Y_tr, 'Training')
plot_state_prediction(states_list, test_prediction, Y_t, 'Testing')
