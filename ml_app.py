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
    st.subheader('Machine Learning')

    #1.유저에게 입력받는다


    gender = st.radio('성별을 선택하세요',['남','여'])
    if gender == '남자':
        gender = 1
    else :
        gender = 0

    age = st.number_input('나이를 입력하세요',0,120)
    annual_salary = st.number_input('연봉을 입력하세요',0)
    credit_card_debt = st.number_input('카드빚을 입력하세요',0)
    net_worth = st.number_input('순자산을 입력하세요',0)

    #2. 예측한다
    
    #2-1 모델불러오기
    model = tensorflow.keras.models.load_model('ann_model.h5')
    #2-2 스케일링
    new_data = np.array([gender,age,annual_salary,credit_card_debt,net_worth])
    new_data = new_data.reshape(1,-1)
     
    mn = joblib.load('mn.pkl')
    data_scaled = mn.transform(new_data)
    y_pred = model.predict(data_scaled)
    

    data_scaled = mn.inverse_transform(y_pred)
    btn = st.button('결과보기')
    if btn :
        st.write('당신이 구매 가능한 차량의 금액대는 {:,.1f}달러입니다'.format(data_scaled[0,0]))


    
    