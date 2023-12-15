import streamlit as st

#메인로직 및 공통함수점검 
import app_common as app_common

#필요한 함수들
import re
from datetime import datetime
import locale

#EDA
import pandas as pd
import plotly.express as px

def page_main(): 
    st.subheader("준비작업 #1 : Chat Data 데이터 클린징 ")  
    app_common.set_side_fileinfo()
    st.info("처리내용: 화자별 데이터셋 분리, 시간정보 일시정보 변환")

    # 함수호출전 초기화 
    st.session_state.speaker_sentiment_sentence = None
    if "FileData" in st.session_state:
        if st.session_state.FileData is not None:
            speaker_sentiment_sentence, fulltext = get_conversation_data(st.session_state.FileData)
            
            col1, col2 = st.columns([3, 7])
            with col1 :
                st.subheader("데이터 구조분리 by Speaker")
                st.write(speaker_sentiment_sentence)
            with col2 : 
                st.subheader("데이터 구조분리 by Date")
                get_base_datainfo(speaker_sentiment_sentence)


def get_conversation_data(conversation_data):
    speaker_sentiment_sentence={}
    current_timestamp = None
    
    # Set the locale to Korean
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
    iLen = len(conversation_data)

    #st.write(f"Data Line: {iLen}")
    #st.write(conversation_data)
    fulltext = ''
    icnt = 0
    for line in conversation_data:
        icnt += 1
        line = line.strip()
        #st.info(str(icnt) + line) 
        if not line:
            continue
        
        #st.error(line) 
        #일시패턴 구하기 
        timestamp = extract_timestamp(line)
        if timestamp:
            #st.warning(str(icnt) + timestamp)
            current_timestamp = datetime.strptime(timestamp, "%Y년 %m월 %d일 %A")
            continue
        
        #st.warning(str(icnt) + line)  
        #대화패턴 검토
        parts = line.split('] [')
        if len(parts) < 2:
            #st.warning(str(icnt) + line) 
            continue
        
        
        speaker = parts[0][1:]
        sentence = ' '.join(parts[1:])

        # 카톡패턴 적용 : 정규표현식 패턴으로 사람, 일시, 대화를 추출 ("person", "timestamp", and "text")
        pattern = r'\[(.*?)\] \[(.*?)\] (.*)'
        matches = re.findall(pattern, line)

        if matches:
            person, timestamp, text = matches[0]
            fulltext = fulltext + " " + text

            #st.error(str(icnt) + line) 
            # speaker_sentiment_sentence[person].append({
            #     "date" : current_timestamp.strftime("%Y년 %m월 %d일"),
            #     "timestamp": timestamp,
            #     "text": text
            # })
           
            if person in speaker_sentiment_sentence :
                #st.warning('**' + str(icnt) + 'add' + line) 
                speaker_sentiment_sentence[person].append({
                    "date" : current_timestamp.strftime("%Y년 %m월 %d일"),
                    "timestamp": timestamp,
                    "text": text
                })
            else:
                #st.warning('***' + str(icnt) + ' new ' + line) 
                speaker_sentiment_sentence[person] = [{
                    "date" : current_timestamp.strftime("%Y년 %m월 %d일"),
                    "timestamp": timestamp,
                    "text": text
                }]
            #st.error(f"* {str(icnt)} {speaker_sentiment_sentence}")
        else :
            pass
            #st.warning(str(icnt) + line)     
        #st.success(str(icnt) + line) 
    st.session_state.speaker_sentiment_sentence = speaker_sentiment_sentence
    st.session_state.chatmsg_fulltext = fulltext

    #st.write(str(icnt) + line) 
    #st.write(speaker_sentiment_sentence)
    return speaker_sentiment_sentence, fulltext

# 이 함수는 주어진 데이터를 처리하여 대화자와 날짜별 대화 건수를 구합니다.
def get_base_datainfo(data):
    data_for_chart = []
    # 대화자별 대화 건수 처리 : 데이터를 대화자와 대화 내용으로 분류한 뒤, 날짜별 대화 건수를 계산합니다.
    for speaker, conversations in data.items():        
        #사람별 대화수 초기화 
        conversation_count = 0
        currentdate = conversations[0]["date"]
        #st.info(speaker + ":" + currentdate)

        #사람부터 
        for conversation in conversations:
            #날짜가 달라지면 
            if  conversation["date"] != currentdate:
                # 지금까지 기록한것을 찍어라
                data_for_chart.append({
                    "speaker": speaker,
                    "date": currentdate,
                    "conversation_count": conversation_count,
                })    

                #닐찌가 바뀐 것
                conversation_count = 1
                currentdate = conversation["date"]   
            else :
                #같은날짜면                
                conversation_count += 1
                
        data_for_chart.append({
            "speaker": speaker,
            "date": currentdate,
            "conversation_count": conversation_count,
        })  

    # 대화자별 대화 건수를 저장하는 데이터 리스트 출력 (개발 중 디버깅용)
    st.write(data_for_chart)

    # Pandas DataFrame 생성 : 데이터를 Pandas DataFrame으로 변환합니다.
    df = pd.DataFrame(data_for_chart)
    
    # 색상 테마 정의 및 Plotly를 사용하여 막대 차트 생성 :색상 테마를 정의하고 Plotly를 사용하여 날짜별 대화 건수를 막대 차트로 생성합니다.
    color_theme = ["blue", "red", "green", "orange", "purple", "yellow"]
    fig = px.bar(df, x="date", y="conversation_count", color="speaker", color_discrete_sequence=color_theme)

    # 차트 커스터마이징 : 차트의 제목, 축 레이블 등을 커스터마이징합니다.
    fig.update_layout(
        title="일자별 대화수",
        xaxis_title="대화일자",
        yaxis_title="대화수",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(r=0, t=0, b=0, l=0),
    )

    # Streamlit을 사용하여 차트 표시 : 차트를 Streamlit 컨테이너에 표시합니다.
    with st.container():
        # Plotly 차트를 Streamlit으로 표시 (툴바는 비활성화, 컨테이너 너비를 사용자 화면에 맞게 설정)
        st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

# 카카오톡에 --------------- 2022년 5월 15일 일요일 -------------- 이런 패턴 정리 
def extract_timestamp(line):
    pattern = r'[-─]+ (\d+년 \d+월 \d+일 \w요일) [-─]+'
    matches = re.findall(pattern, line)
    if matches:
        return matches[0]

#최초실행시 호출 
if __name__ == "__main__":
    if app_common.session_check():
        page_main()
    else :
        st.title("메인페이지로 돌아가세요")