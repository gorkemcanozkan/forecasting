import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from fbprophet import Prophet

st.title("Forecasting")
df=pd.read_excel(st.file_uploader("Dosyanızı Import Ediniz"))
donem=st.sidebar.text_input("Kaç Dönem Tahmin Etmek İstiyorsunuz?")
donem=int(donem)
select=st.sidebar.radio("İyimser - Orta - Kötümser Tahmin", ("İyimser","Ortalama","Kötümser"))
if select=="İyimser":
    select_="yhat_upper"
elif select=="Ortalama":
    select_="yhat"
elif select=="Kötümser":
    select_="yhat_lower"
if st.sidebar.button("Tahminle"):
    col_list = df.columns[1:]
    ff = [
    
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true+=0.001
        return np.mean(np.abs((y_pred - y_true) /np.abs(y_true))) * 100
    
    for item in col_list:
        df = df.rename(columns={item: "y"})
        df1 = df[['ds', 'y']]
        m = Prophet()
        m.fit(df1)
        future = m.make_future_dataframe(periods=donem, freq='M')
        forecast = m.predict(future)
        ff.append(forecast[select_].tail(donem))
        df.drop(columns=["y"], inplace=True)
        st.write(f'MAPE for {item}: ',mean_absolute_percentage_error(df1.y,forecast.yhat))
    x=pd.DataFrame(ff).T
    x.columns=col_list
    x[x<0]=0
    st.write(x)
