import streamlit as st
from PIL import Image
import pandas as pd
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


from eda_app import run_eda_app
from ml_app import run_ml_app


def main():
    st.title('자동차 가격 예측')

    menu = ['Home','EDA','ML']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Home':
        st.write('이 앱은 고객 데이터를 바탕으로 자동차 구매 예측을 하는 인공지능 앱입니다')
        st.write('좌측 메뉴를 선택하세요')
    elif choice == 'EDA':
        run_eda_app() 
    elif choice == 'ML':
        run_ml_app()







if __name__ == '__main__':
    main()