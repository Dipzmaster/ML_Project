# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:13:58 2022

@author: HP
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start_date = '2016-01-01'
end_date = '2022-02-02'

st.title('Stock Prediction')

user_input = st.text_input('Enter Stock', 'TSLA')
df = data.DataReader(user_input, 'yahoo', start_date, end_date)

#Describe data
st.subheader('Data from 2016 to 2022')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 and 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


data_t = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_tst = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

data_t_array = scaler.fit_transform(data_t)



model = load_model('keras_model.h5')

past_100_days = data_t.tail(100)

final_df = past_100_days.append(data_tst, ignore_index =True)

input_data = scaler.fit_transform(final_df)

x_tst = []
y_tst = []

for i in range(100, input_data.shape[0]):
    x_tst.append(input_data[i-100:i])
    y_tst.append(input_data[i,0])

x_tst, y_tst = np.array(x_tst), np.array(y_tst)

y_pred = model.predict(x_tst)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_pred = y_pred * scale_factor
y_tst = y_tst * scale_factor


st.subheader('Pred vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_tst, 'b', label = 'Original Price')
plt.plot(y_pred, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
