import streamlit as st

def session_check():   
   if "shared" not in st.session_state:      
      return False
   return True

# 파일현황
def set_side_fileinfo():
    if "uploaded_file" not in st.session_state:   
      st.session_state.uploaded_file = None

    uploaded_file = st.session_state.uploaded_file
    if uploaded_file is not None:
        with st.sidebar:
            #st.caption("분석대상파일")
            sInfo =  '- File = ' + uploaded_file.name + " \n"
            sInfo += '- Type = ' + uploaded_file.type + " \n"
            sInfo += '- Size = ' +  str(round(uploaded_file.size / 1024/1024, 3) ) +  " MByte"
            st.info(sInfo)

