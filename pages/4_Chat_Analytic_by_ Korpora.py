import streamlit as st

#메인로직 및 공통함수점검 
import app_common as app_common

#필요한 모듈
import os
import json

#DataSet
from Korpora import Korpora

#EDA
import pandas as pd
import plotly.express as px

def page_main():
    st.subheader("분석작업 #4 : Korpora 데이터셋 확보하기 (향후진행)")  
    app_common.set_side_fileinfo()
    st.info("처리내용: Korpora 협오데이터셋 확보하여 데이터 분석하기 하기")
    
    # 클린징 값이 들어왔으면 아래 로직 수행 (클린징은 1_Data_Cleansing.py 로직에서 수행)
    if "speaker_sentiment_sentence" in st.session_state:
        if st.session_state.speaker_sentiment_sentence is not None:
            speaker_sentiment_sentence = st.session_state.speaker_sentiment_sentence

    # 한국어말뭉치 확보하기
    korpora_list = Korpora.corpus_list() # 전체한국어말뭉치데이터 
    with st.expander("필요한 Korpora 학습데이터 추가하기") :
        get_Korpora_info(korpora_list)

    korpora_local_list = get_exists_dataset(korpora_list) # 내려받은 한국어말뭉치데이터 (#korean_hate_speech)
    selected_dataset = st.selectbox("보유하고 있는 데이터셋을 선택하세요", list(korpora_local_list.keys()),get_base_dataset(korpora_local_list, "korean_hate_speech") )
    
    # 협오 데이터확보하기 
    if selected_dataset is not None:
        #st.write(selected_dataset)
        pass

def get_Korpora_info(korpora_list):
    korpora_combolist = {}
    for key, value in korpora_list.items():
        value_txt = f"[{key}] {value}"
        korpora_combolist[value_txt] = key

    selected_dataset = st.radio("Korpora 데이터셋을 선택하세요", list(korpora_combolist.keys()), 2)
    if selected_dataset is not None:
        korporaSet = korpora_combolist[selected_dataset]
        get_Korpora_download(korporaSet)

    return korpora_list

#
def get_Korpora_download(korporaSet):
    if st.button(korporaSet + ' 말뭉치다운로드') :
        with st.spinner("DownLoading..."):
            Korpora.fetch(korporaSet, root_dir="./model/Korpora")
        st.write(korporaSet + " 다운로드 완료")


# 보유하고 있는 다운받은 데이터 확인하기 
def get_exists_dataset(korpora_list):
    root_dir = "./model/Korpora"
    folder_names = os.listdir(root_dir)
    existdata = {}
    for key, value in korpora_list.items():
        if key in folder_names:
            existdata[key] = value
    return existdata

def get_base_dataset(korpora_local_list, key_name_to_find):
    #key_name_to_find = "kcbert"

    if key_name_to_find in korpora_local_list:
        index = list(korpora_local_list.keys()).index(key_name_to_find)
    else:
       index = 0 #없음
    
    return index

#최초실행시 호출 
if __name__ == "__main__":
    if app_common.session_check():
        page_main()
    else :
        st.title("메인페이지로 돌아가세요")