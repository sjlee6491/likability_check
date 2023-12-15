import streamlit as st

#메인로직 및 공통함수점검 
import app_common as app_common

#공통함수
import time


def page_main():
    st.subheader("About .... ")  
    app_common.set_side_fileinfo()
    st.info("Program : 2023.08.02 by Ryan /이승진 (toi300 member, 원묵중학교), sjlee6491@naver.com")
    
    # 클린징 값이 들어왔으면 아래 로직 수행 (클린징은 1_Data_Cleansing.py 로직에서 수행)
    if "speaker_sentiment_sentence" in st.session_state:
        if st.session_state.speaker_sentiment_sentence is not None:
            speaker_sentiment_sentence = st.session_state.speaker_sentiment_sentence

    set_ui_devview()
    set_ui_devnext()


def set_ui_devview():
    st.caption("[개발방향]")
    strText =  ""
    strText += f"- 방향 : 채팅메세지를 분석해서 대화상대의 사용언어를 통해 나에대한 호감도, 혐오도 등의 성향을 파악하는 기능 구현" + "\n"
    strText += f"- 적용 : pandas, numpy, spyci, keras 등을 활용..." + "\n"
    strText += f"- 자료 : KNU 한국어감성사전, 한국어감성사전 추가사전 (요즘 채팅말 일부추가) ..." + "\n"
    st.caption(strText)

def set_ui_devnext():
    st.caption("[향후계획]")
    strText =  ""
    strText += f"- (미정) 원문 데이터 클린징 강화 방법 검토 ..." + "\n"
    strText += f"- (미정) Korpora 데이터 활용 평가자료 응용적용 검토" + "\n"
    strText += f"- (미정) 다자간 대화 관계성 그래프 시각화 방법 검토" + "\n"
    
    st.caption(strText)


#test 1       
def set_ui_test1():
    empty_container = st.empty()  # 빈 컨테이너 생성    
    for i in range(21):
        smsg = f"test {i}"
        with empty_container.container():
            st.write(smsg)
        time.sleep(0.4)
    # empty_container.empty()

#test 2       
def set_ui_test2():
    sdata = st.radio("선택하세요", ('Comedy', 'Drama'), horizontal=True, label_visibility='hidden')


#최초실행시 호출 
if __name__ == "__main__":
    if app_common.session_check():
        page_main()
    else :
        st.title("메인페이지로 돌아가세요")