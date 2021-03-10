import streamlit as st
from PIL import Image
import pandas as pd
from PyPDF2 import PdfFileReader # pdf파일 읽어오는 라이브러리
import os
import h5py
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
import pickle
import joblib

def run_ml_app():
    st.subheader('ML 화면입니다')

    model = tensorflow.keras.models.load_model('ann_model.h5')

    new_data = np.array([0,38,90000,2000,500000])
    new_data = new_data.reshape(1,-1)
     
    mn = joblib.load('mn.pkl')
    data_scaled = mn.transform(new_data)
    y_pred = model.predict(data_scaled)
    

    data_scaled = mn.inverse_transform(y_pred)
    st.write(data_scaled)


    
    