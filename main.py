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
if st.button("Tahminle"):
    col_list = df.columns[1:]
    ff = []
    for item in col_list:
        df = df.rename(columns={item: "y"})
        df1 = df[['ds', 'y']]
        m = Prophet()
        m.fit(df1)
        future = m.make_future_dataframe(periods=donem, freq='M')
        forecast = m.predict(future)
        ff.append(forecast[select_].tail(donem))
        df.drop(columns=["y"], inplace=True)
    x=pd.DataFrame(ff).T
    x.columns=col_list
    x[x<0]=0
    st.write(x)
