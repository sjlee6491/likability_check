import streamlit as st

#메인로직 및 공통함수점검 
import app_common as app_common

#필요한 모듈
import os
import json

#EDA
import pandas as pd
import plotly.express as px

def page_main(): 
    st.subheader("분석작업 #3 : KNU 한국어 감성사전으로 대화분석 해보기")  
    app_common.set_side_fileinfo()
    st.info("처리내용: KNU 한국어감성사전 추가확장 및 감성사전기반 개별문장 총점, 유형평점, 사용빈도 계산")
    

    # 클린징 값이 들어왔으면 아래 로직 수행 (클린징은 1_Data_Cleansing.py 로직에서 수행)
    if "speaker_sentiment_sentence" in st.session_state:
        if st.session_state.speaker_sentiment_sentence is not None:
            speaker_sentiment_sentence = st.session_state.speaker_sentiment_sentence
            sentiment_dict_user = get_knu_add_user_dictionary()

            for speaker, entries in speaker_sentiment_sentence.items():
                for entry in entries:
                    text_value = entry["text"]
                    # Calculate the score for the text using KNU sentiment dictionary
                    score_user = calculate_sentence_sentiment(text_value, sentiment_dict_user)
                    entry["knu_user_sum"] = score_user[0] #문장점수 
                    entry["knu_user_cntsum"] = score_user[1] #사용빈도수                  
                    entry["knu_user_score"] = score_user[2] #문장상세점수
                    entry["knu_user_count"] = score_user[3] #사용상세빈도

            #st.write(speaker_sentiment_sentence)
            #st.write(speaker_sentiment_sentence)
            st.write("KNU+ 감성사전 데이터정보")
            sLen = str(len(sentiment_dict_user))
            st.text_area(f'총 사전데이터 건수 : {sLen}', sentiment_dict_user, height=200, )

            df = get_dataframe(speaker_sentiment_sentence)
            st.write("대화별: KNU+ 감성사전 문장평가")
            st.write(df)

            result_df = get_group_info(df)
            # 결과 DataFrame 출력
            st.write("대화자별 : 감성사전 기반 데이터 합계 평가")
            st.write(result_df)

#dp정보
def get_group_info(df):
    def sum_scores(df):
        result = {key: 0 for key in sentiment_scores}
        #results = {}
        values = 0
        for knuscore in df.iloc:
            result_dict = knuscore["KNU+Score"] 
            for key, value in result_dict.items():
                keynm = str(key)
                values = int(value)
                result[keynm] += values                  
        return result
    
    def sum_counts(df):
        result = {key: 0 for key in sentiment_scores}
        #results = {}
        values = 0
        for result_dict in df["KNU+Counts"].iloc:
            for key, value in result_dict.items():
                keynm = str(key)
                values = int(value)
                result[keynm] += values                  
        return result

    # DataFrame에 적용
    df_grouped1 = df.groupby("Speaker").apply(sum_scores).reset_index()
    df_grouped1 = df_grouped1.rename(columns={0: "KNU+Score"})
    df_grouped2 = df.groupby("Speaker").apply(sum_counts).reset_index()
    df_grouped2 = df_grouped2.rename(columns={0: "KNU+Counts"})
    df_additional_agg  = df.groupby("Speaker").agg({
        "Text": "count",
        "KNU+ScoreSum": "sum",
        "KNU+CountSum": "sum"
        }).reset_index()
    
    result_df = df_grouped1.merge(df_additional_agg, on="Speaker", suffixes=("_grouped", "_agg"))
    result_df = df_grouped2.merge(result_df, on="Speaker", suffixes=("_grouped", "_agg"))

    return result_df
    
@st.cache_data
def get_knu_dictionary():
    # KNU 한국어 감성사전 불러오기
    sentiment_dict = load_sentiment_dict("SentiWord_info.json")

    return sentiment_dict

@st.cache_data
def get_knu_add_user_dictionary():
    # get_dictionary에서 불러온 기존 감성 사전
    sentiment_dict = get_knu_dictionary()
    
    # 요즘 애들 쓰는말이 반영되어 있지 않아서서, 추가함 
    user_sentiment_dict = load_sentiment_dict("SentiWord_info_user.json")

    # 기존 감성 사전에 추가 사전을 합침
    sentiment_dict.update(user_sentiment_dict)

    return sentiment_dict

# 분석데이터를 데이터프레임에 넣기 
def get_dataframe(speaker_sentiment_sentence):
    speakers = []
    dates = []
    timestamps = []
    texts = []
    knu_user_sum = []
    knu_user_cntsum = []
    knu_user_score = []
    knu_user_count = []

    for speaker, entries in speaker_sentiment_sentence.items():
        for entry in entries:
            speakers.append(speaker)
            dates.append(entry["date"])
            timestamps.append(entry["timestamp"])
            texts.append(entry["text"])      
            knu_user_sum.append(entry["knu_user_sum"])
            knu_user_cntsum.append(entry["knu_user_cntsum"])
            knu_user_score.append(entry["knu_user_score"])
            knu_user_count.append(entry["knu_user_count"])

    # 리스트들을 딕셔너리로 변환하여 데이터프레임 생성
    df_data = {
        "Speaker": speakers,
        "Date": dates,
        "Timestamp": timestamps,
        "Text": texts,
        "KNU+ScoreSum": knu_user_sum,
        "KNU+CountSum": knu_user_cntsum,
        "KNU+Score": knu_user_score,
        "KNU+Counts": knu_user_count
    }

    df = pd.DataFrame(df_data)
    return df


# 감성 사전 불러오기 함수
def load_sentiment_dict(filename):
    folder = 'model'
    file_path = os.path.join(folder, filename)  # 파일 경로 생성
    with open(file_path, 'r', encoding='utf-8') as f:
        sentiment_data = json.load(f)

    sentiment_dict = {item['word']: int(item['polarity']) for item in sentiment_data}
    return sentiment_dict       

# 문장에서 감성 점수 계산 함수
# 각 성향별 점수를 나타내는 딕셔너리
sentiment_scores = {
    "매우부정": -2,
    "부정": -1,
    "중립": 0,
    "긍정": 1,
    "매우긍정": 2
}
# "-2:매우 부정, -1:부정, 0:중립 or Unkwon, 1:긍정, 2:매우 긍정"
def calculate_sentence_sentiment(sentence, sentiment_dict):
    words = sentence.split()
    sentiment_total_score = 0
    sentiment_total_count = 0
    sentiment_score = {score_category: 0 for score_category in sentiment_scores.keys()}
    sentiment_count = {score_category: 0 for score_category in sentiment_scores.keys()}

    for word in words:
        if word in sentiment_dict:
            score = sentiment_dict[word]
            sentiment_total_score += score #누적점수
            sentiment_total_count += 1 #빈도
            if score in sentiment_scores.values():
                score_category = get_score_category(score)
                score_sum = sentiment_score[score_category]
                freq      = sentiment_count[score_category]
                #st.write(freq)
                sentiment_score[score_category] = score_sum + score
                sentiment_count[score_category] = freq + 1
  
    sentiment_score_last = (sentiment_total_score, sentiment_total_count, sentiment_score, sentiment_count)
    return sentiment_score_last

# 감성 점수에 따른 카테고리 반환 함수
def get_score_category(score):
    for category, value in sentiment_scores.items():
        if score == value:
            return category


#최초실행시 호출 
if __name__ == "__main__":
    if app_common.session_check():
        page_main()
    else :
        st.title("메인페이지로 돌아가세요")