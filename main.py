import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from fbprophet import Prophet

st.title("Forecasting")
df=pd.read_excel(st.file_uploader("Dosyanızı Import Ediniz"))
k_adi=st.sidebar.text_input("Kullanıcı Adı")
sifre=st.sidebar.text_input("Şifre")
donem=st.sidebar.text_input("Kaç Dönem Tahmin Etmek İstiyorsunuz?")
donem=int(donem)
select=st.sidebar.radio("İyimser - Orta - Kötümser Tahmin", ("İyimser","Ortalama","Kötümser"))
if select=="İyimser":
    select_="yhat_upper"
elif select=="Ortalama":
    select_="yhat"
elif select=="Kötümser":
    select_="yhat_lower"

if k_adi=="gizemsenol" and sifre=="publicis.1234@2021":
    col_list = df.columns[1:]
    ff = []
    for item in col_list:
        df = df.rename(columns={item: "y"})
        df1 = df[['ds', 'y']]
        m = Prophet()
        m.fit(df1)
        future = m.make_future_dataframe(periods=donem, freq='M')
        forecast = m.predict(future)
        ff.append(forecast[select_].tail(donem).values[0])
        df.drop(columns=["y"], inplace=True)
    st.write(pd.DataFrame(ff).T)
