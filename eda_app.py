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

    choice_list = df.columns
    choice_list = list(choice_list)
    selected_columns = st.multiselect('컬럼을 선택하세요',choice_list)
    if len(selected_columns) != 0 :
        st.dataframe(df[selected_columns])
    else:
        st.write('선택한 컬럼이 없습니다.')

    #상관계수를 화면에 보여주도록 만듭니다
    #멀티셀렉트에 컬럼명을 보여주고,
    #해당 컬럼들에 대한 상관계수를 보여주세요
    #단, 컬럼들은 숫자 컬럼들만 멀티셀렉트에 나타나야 합니다
    
    corr_columns = df.columns[df.dtypes != object]
    selected_corr = st.multiselect('상관계수 컬럼 선택',corr_columns)
    
    if len(selected_corr) != 0 :
        st.dataframe(df[selected_corr].corr())
        
        fig = sns.pairplot(data = df[selected_corr])
        st.pyplot(fig)

    else:
        st.write('선택한 컬럼이 없습니다.')

    #컬럼 하나를 선택하면, 해당 컬럼의 min과 max에 해당하는 사람의 데이터를 보여줘라
    selected_col2 = st.selectbox('컬럼을 선택하세요',corr_columns)
    if selected_col2 is not None:
        st.write('최대')    
        st.dataframe(df.loc[df[selected_col2] == df[selected_col2].max(),])
        st.write('최소')    
        st.dataframe(df.loc[df[selected_col2] == df[selected_col2].min(),])

    #고객이름을 검색할 수 있는 기능을 개발
    word = st.text_input('이름을 입력하세요')

    result = df.loc[df['Customer Name'].str.contains(word, case = False),]

    st.dataframe(result)


    

    # search = st.text_input('이름을 입력하세요')
    # st.dataframe(df.loc[df['Customer Name'],])

    


    


    
    