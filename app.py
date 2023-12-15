import streamlit as st
import app_common as app_common

st.set_page_config(layout="wide", page_title="Chat Data 분석")

#페이지별 공유정보 
if "shared" not in st.session_state:
   st.session_state["shared"] = False

#시작함수
def main():
    st.title("Chat Data 분석 App: by Ryan") 
    st.sidebar.subheader("Chat Data 분석 App")
    
    st.subheader("기본작업 #0 : Chat Data 데이터셋 불러오기")  
    st.info("카카오톡 채팅내역을 txt로 저장해서 파일업로드시 호감도 정보를 파악해보고자 합니다")  
       
    # set_sidemenu() 함수를 호출하여 data를 얻습니다.
    uploaded_file = file_uploader_widget()
    result_data = process_uploaded_file(uploaded_file)
    
    # 새로불러올 데이터가 존재하면 
    if result_data is not None:   
        st.session_state.uploaded_file = uploaded_file
        st.success(uploaded_file.name + " 새로운 데이터를 불러옴")        
        st.session_state.FileData = result_data
        #st.write(result_data)
    elif "FileData" in st.session_state:
        if st.session_state.FileData is not None:
            st.success("기존 데이터를 불러옴")
            uploaded_file = st.session_state.uploaded_file
            result_data = st.session_state.FileData
            #st.write(result_data)
    else :
        st.session_state.uploaded_file = None
        st.session_state.FileData = None
    
    #main info 
    if st.session_state.FileData is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw DataSet Structure")
            st.caption("- 불러온 데이터에 대해서 Line 기준 Raw 데이터 분리처리")
            st.write(result_data)    
        with col2:
            st.subheader("Raw DataSet Grid")
            st.caption("- 불러온 데이터에 대해서 Line 기준 데이터프레임 처리")
            df = st.dataframe(result_data, use_container_width=True, hide_index=False)  

    app_common.set_side_fileinfo()

# 파일현황
def set_side_fileinfo():
    uploaded_file = st.session_state.uploaded_file
    if uploaded_file is not None:
        with st.sidebar:
            st.caption("분석대상파일: " + uploaded_file.name)      
  
def process_uploaded_file(uploaded_file):
    return_contents = None

    if uploaded_file is not None:
        # To read file as bytes:
        file_contents = uploaded_file.read()
        # Decode bytes to string
        conversation_data = file_contents.decode('utf-8').splitlines()

        return_contents = conversation_data

    # conversation_data 또는 None 반환
    return return_contents

# 위젯을 사용하여 파일을 업로드하는 함수
def file_uploader_widget():
    uploaded_file = st.file_uploader("☞ 분석할 Chat Data을 올려주세요", type=["txt"])
    return uploaded_file

def main_call():
    s_filename = None
    if "uploadfilename" in st.session_state : 
        s_filename = st.session_state.uploadfilename
    return s_filename

#최초실행시 호출 
if __name__ == "__main__":
    main()
