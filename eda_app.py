import streamlit as st
from PIL import Image
import pandas as pd
from PyPDF2 import PdfFileReader # pdf파일 읽어오는 라이브러리
import os
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') #서버에서 화면에 표시하기 위해서 필요
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def run_eda_app():
    st.subheader('EDA 화면입니다')
    df = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')

    radio_menu = ['데이터프레임','통계치']
    selected_radio = st.radio('선택하세요',radio_menu)
    if selected_radio == '데이터프레임':
        st.dataframe(df)
    elif selected_radio == '통계치':
        st.dataframe(df.describe())

    


    
    