from models import get_model
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
def fit_model(input_seq, output_seq, in_win, out_win, num_epochs, batchsize, train_mode, model_name):
    n_regions =  input_seq.shape[-1]
    #print(f"n_regions: {n_regions}")
    model = get_model(model_name, in_win, out_win, n_regions, dropout=0, training_t=True)

    # design network
    my_adam = optimizers.Adam(lr=0.0001, decay=0.0)
    filepath = "./Training/best_weights_" + model_name +  ".h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    
    earlystopping = EarlyStopping(monitor='loss', patience=50, verbose=1)
    callbacks_list = [checkpoint,earlystopping]
    #print(model.summary())
    # fit network
    model.compile(optimizer=my_adam,loss='mean_squared_error', run_eagerly=True)
    if train_mode:
        train_history = model.fit(input_seq, output_seq, epochs=num_epochs, batch_size=batchsize, verbose=0, callbacks=callbacks_list) 
        np.save('./Training/Train_Loss_'+model_name + '.npy',train_history.history['loss'])

    # load model 
    model.load_weights(filepath)
    return model

def forecast_model(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = np.expand_dims(X[:,:],0)
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    #print(forecast.shape)
    #print([x for x in forecast[0, :, :]])
    #return [x for x in forecast[0, :, :]]
    return forecast

def make_forecasts(model, n_batch, data, in_win, out_win):
    forecasts = np.zeros(shape=(len(data), out_win, data.shape[2]))
    for i in range(len(data)):
        X = data[i, 0:in_win, :]
        # make forecast
        forecast = forecast_model(model, X, n_batch)
        forecasts[i,:,:] = forecast
    return forecasts

    
def make_forecasts_with_fb(model, n_batch, data, in_win, out_win):
    #print(data.shape) # None, in_win, n_counties
    forecasts = np.zeros(shape=(len(data), 4, data.shape[2]))
    for i in range(len(data)):
        X = np.copy(data[i, :, :]) # in_win, n_counties
        for j in range(4):
            # make forecast
            forecast = forecast_model(model, X, n_batch)
            forecasts[i,j,:] = forecast
            #print(X[:,0:10])
            X[0:in_win-1,:] = X[1:,:]
            X[in_win-1,:] = forecast[0,0,:]
            #print(X[:,0:10])
            #print(var)
    return forecasts

def eval_forecasts(true, pred, in_win, out_win):
    #print(f"true data: {true}")
    #print(f"pred data: {pred}")
    rmse_out = np.zeros(shape=(out_win))
    mape_out = np.zeros(shape=(out_win))
    mae_out = np.zeros(shape=(out_win))
    for i in range(out_win):
        actual = true[:,i,:] 
        forecast = pred[:,i,:]
        
        RMSE = np.sqrt(mean_squared_error(actual, forecast))
        MAPE = np.mean((abs(actual - forecast) / (abs(actual)+1)))
        MAE = np.mean(abs(actual - forecast))
        
        #print('t+%d RMSE: %f' % ((i+1), RMSE))    
        #print('t+%d MAPE: %f' % ((i+1), MAPE))    
        #print('t+%d MAE: %f' % ((i+1), MAE))    
       
        mape_out[i] = MAPE
        mae_out[i] = MAE
        rmse_out[i] = RMSE
    return rmse_out, mape_out, mae_out

