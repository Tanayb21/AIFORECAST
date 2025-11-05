#This is visually designed APP 
# TO RUN , FIRST READ THE README FILE  >_<

#Imports
import streamlit as st
from prophet import Prophet
import seaborn as sb    
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
from sklearn.metrics import mean_absolute_error , mean_squared_error
import numpy as np

#TITLE FOR WEBPAGE
st.set_page_config(page_title="ANJ ASSIGNMENT", layout="wide")

st.title("AI PROJECT ASSIGNMENT ANJ")


#UPLOADER FOR DATA

fileupload =  st.file_uploader("Upload a CSV File ",type=['csv'])

#A condition if the file is uploaded 
if fileupload:
    df= pl.read_csv(fileupload)
else:
    st.info("Upload the file first , Make sure it's csv !")

if not fileupload:
    st.stop()

#Once uploaded , it will make initial (primitive) graph 
st.subheader("Data Preview (First 5 rows)")
st.dataframe(df.head())
pandas_df = df.to_pandas()
st.subheader("Note: It may take time to process and make graphs depending on system config")
sb.set(style='whitegrid')

cont=st.empty()
with cont.container():
    with st.spinner("Generating graph... Please wait ⏳"):

        figure , ax = plt.subplots(figsize=(5,4))
        sb.lineplot(data=pandas_df , x="datetime" , y="value" , color='c', ax=ax)
        plt.title("Primitive Trend ")
        st.pyplot(figure)

cont.success("Graph made successfully")
st.pyplot(figure)

#Model training starts here 


st.subheader("Training the model on uploaded data ")

#radio buttons for choosing time to predict next values

choose_period =st.radio("Select the period of forecast ",['6 Months','1 year' ,'2 years'])

pandas_df = pandas_df.rename(columns={'datetime':"ds" , "value":"y"})
pandas_df['ds'] = pd.to_datetime(pandas_df['ds'])

Pred_model = Prophet(daily_seasonality=True , yearly_seasonality=True)
Pred_model.fit(pandas_df)



if choose_period =='6 Months':
    future_pred = Pred_model.make_future_dataframe(periods = 24*30*6 , freq="H")
    
elif choose_period =='1 year':
    future_pred = Pred_model.make_future_dataframe(periods = 24*365*1 , freq="H")

elif  choose_period =='2 years':
        future_pred = Pred_model.make_future_dataframe(periods = 24*365*2 , freq="H")


#training the model on the given dataset 

forecast = Pred_model.predict(future_pred)

#Plotting the data with comparison to previous data

st.subheader("Visualization of Forecasted data")

figure2 , ax2 = plt.subplots(figsize=(14,6))
sb.lineplot(data=pandas_df , x="ds" , y="y" , color='b', ax=ax2 ,label="Actual Data")
sb.lineplot(data=forecast , x="ds" , y="yhat" , color='c', ax=ax2 ,label="Forecasted Data")
plt.title(f"Signal Forecast ({choose_period})")
plt.xlabel("Date")
plt.ylabel("Signal Value")
plt.legend()
st.pyplot(figure2)

#downlaod forecasted data in CSV Format.

st.subheader("Download forecast (CSV)")
csv= forecast.to_csv(index=False ).encode('utf-8')
st.download_button("Download forecasted data in CSV",csv)


#MAE and RMSE

merged= pd.merge(pandas_df , forecast[['ds','yhat']], on='ds', how='inner')

mae= mean_absolute_error(merged['y'],merged['yhat'])
rmse =  np.sqrt(mean_squared_error(merged['y'],merged['yhat']))

st.subheader("ERRORS ")
st.write("MEAN ABSOLUTE ERROR ", mae)
st.write("MEAN SQUARE ERROR ", rmse)
