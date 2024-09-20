import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st

st.title('Stock trend Prediction')

stock_symbol = st.text_input('Enter Stock Ticker',"GAIL.NS")

#last 5 years data with interval of 1 day
data = yf.download(stock_symbol,period='5y',interval='1d')

#Describing Data
st.subheader('Data from 5years - till now')
st.write(data.describe())

#visualizations
st.subheader('Opening Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Open)
st.pyplot(fig)

#Moving Averages
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = data.Open.rolling(100).mean()
ma200 = data.Open.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(data.Open)
st.pyplot(fig)


from keras.models import load_model
model = load_model('new_keras_model.h5')

opn = data[['Open']]
ds = opn.values
plt.plot(ds)

#Using MinMaxScaler for normalizing data between 0 & 1
normalizer = MinMaxScaler(feature_range=(0,1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))

ds_new = ds_scaled.tolist()

#Defining test and train data sizes
train_size = int(len(ds_scaled)*0.70)
test_size = len(ds_scaled) - train_size

ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]

#creating dataset in time series for LSTM model 
#X[100,120,140,160,180] : Y[200]
def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)


#Taking 100 days price as one record for training
time_stamp = 100
X_train, y_train = create_ds(ds_train,time_stamp)
X_test, y_test = create_ds(ds_test,time_stamp)


#Predicitng on train and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#Inverse transform to get actual value
train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)


test = np.vstack((train_predict,test_predict))
#Getting the last 100 days records
fut_inp = ds_test[270:]
fut_inp = fut_inp.reshape(1,-1)
fut_inp.shape

tmp_inp = list(fut_inp)

#Creating list of the last 100 data
tmp_inp = tmp_inp[0].tolist()


#Predicting next 30 days price suing the current data
#It will predict in sliding window manner (algorithm) with stride 1
lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(tmp_inp)>100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp=fut_inp.reshape(1,-1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        fut_inp = fut_inp.reshape((1, n_steps,1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


#Creating final data for plotting
final_graph = normalizer.inverse_transform(ds_new).tolist()

#Creating a dummy plane to plot graph one after another
plot_new=np.arange(1,101)
plot_pred=np.arange(101,131)

plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[1136:]))
plt.plot(plot_pred, normalizer.inverse_transform(lst_output))

#Plotting final results with predicted value after 30 Days
st.subheader('Predictions Vs Original')
fig =plt.figure(figsize=(12, 6))
plt.plot(final_graph,)
plt.ylabel("Price")
plt.xlabel("Time")
plt.title("{0} prediction of next month open".format(stock_symbol))
plt.axhline(y=final_graph[len(final_graph)-1], color = 'red', linestyle = ':', label = 'NEXT 30D: {0}'.format(round(float(*final_graph[len(final_graph)-1]),2)))
plt.legend()
plt.plot(fig)

#final Graph