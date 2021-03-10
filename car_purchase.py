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
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def main():
    st.title('자동차 구매예측 프로그램')
    st.subheader('CSV파일을 불러온다')
    df = pd.read_csv('data\Car_Purchasing_Data.csv')
    st.dataframe(df)

    st.subheader('1. 연봉이 가장 높은사람의 이름은?')
    user_input = st.text_input('정답을 입력하세요')
    if user_input == 'Flores, Caldwell U.':
        st.success('정답입니다')
    else :
        st.error('오답입니다')

    st.subheader('2. 나이가 가장 어린 고객의 연봉은 얼마인가?')
    user_input = st.text_input('정답을 입력하세요',key='num2')
    
    st.subheader('데이터 일부분 보기')
    st.dataframe(df.head())

    st.subheader('NaN 데이터 확인하기')
    st.dataframe(df.isna().sum())
     
    st.subheader('데이터 정보확인하기')
    st.write(df.info())

    st.subheader('X값 설정하기')
    df_X = df.iloc[:,3:7+1]
    st.dataframe(df_X)

    # st.subheader('y값 설정하기')
    mn = MinMaxScaler()
    df_X = mn.fit_transform(df_X)

    y = df['Car Purchase Amount']
    y = y.values.reshape(-1,1)
    sc = StandardScaler()
    y = sc.fit_transform(y)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_X,y, test_size = 0.25)


    st.subheader('새로운 고객 데이터 입력하기')
    option_list = ['남자','여자']
    gender = st.selectbox('성별을 입력해주세요', option_list)
    if gender == '남자':
        gender = 0
    else:
        gender = 1
    age = st.number_input('나이를 입력하세요')
    annual_salary = st.number_input('연봉을 입력하세요')
    debt = st.number_input('카드 빚을 입력하세요')
    worth = st.number_input('순자산을 입력하세요')
    new_data = np.array([gender,age,annual_salary,debt,worth])
    new_data = new_data.reshape(1,-1)

    model = tf.keras.models.load_model('data/ann_model.h5')
    data_scaled = mn.transform(new_data)

    y_pred = model.predict(data_scaled)

    
    y_pred_origin = sc.inverse_transform(y_pred)
    st.write(y_pred_origin)

     #38 90000 2000 500000
if __name__ == '__main__':
    main()