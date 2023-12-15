# likability_check
호감도분석기_고려대영재원프로젝트

- 프로그램
    호감도분석기_고려대영재원프로젝트
    Chat Data 호감도 분석 App: by Ryan 이승진

- 프로그램 구조
    -- model
       * SentiWord_info.json (KNU 한국어 감성사전)
       * SentiWord_info_user.json (KNU 한국어 감성사전 - Ryan 추가정보/요즘카톡에서 애들이 쓰는말 추가)
    -- pages
       * 1_Data_Cleansing.py (데이터 전처리, 대화자별 데이터셋 분리, 년도처리)
       * 2_Chat_Analytic_by_KNU.py (KNU 한국어 감성 데이터 분류)
    -- sample_data
    * app.py

- 가상환경 구성이해 
    * PC에 있는 Python 라이브러리간 충돌방지를 위해 깨끗한 가상환경을 구성하여 실행함

    * 가상환경 만들기 : python -m venv .venvRyan
    * 가상환경 실행하기 : .venvRyan/Scripts/activate 
    * 가상환경 실행중 : (.venvRyan) ~ 
    * 가상환경 종료하기 : .venvRyan/Scripts/deactivate

- spacy 언어모델
    * 설치방법 : 현재로직은 ko_core_news_lg를 사용중임, 가벼운 로직을 위해서는 ko_core_news_sm 설치추천 
        (소) python -m spacy download ko_core_news_sm
        (중) python -m spacy download ko_core_news_md
        (대) python -m spacy download ko_core_news_lg 
    * 참고자료 : https://spacy.io/models/ko

- 설치방법 (가상환경내)
    * 가상환경 실행중 : (.venvRyan) ~ 

    * 설치에 필요한 프로그램 목록 기록 : requirements.txt  
    * 필요한 프로그램 설치 : pip install -r requirements.txt

- 프로그램 실행방법 (가상환경내)    
    * 가상환경 실행중 : (.venvRyan) ~ 
    * 프로그램 실행 : streamlit run ./app.py

    * 프로그램 실행시 아래와 같이 메세지 나옴
        You can now view your Streamlit app in your browser.

        Local URL: http://localhost:8501
        Network URL: http://192.168.0.6:8501

    * 여기 URL 접속하면 화면 조회됨
    * 터미널 종료하기 : ctrl + x 

    * 가상환경 종료하기 : .venvRyan/Scripts/deactivate.bat

- 배치프로그램 실행방법 (가상환경내)
	* 가상환경 이름이 .venvRyan 가상환경이 없으면 생성, pip 설치후 브라우저 호출 실행됨 : start_venvRyan.bat
	* 단 파이썬은 있어야 함 (처음 설치시 라이브러리 다운로드 설치로 시간소요됨, 다음부터는 실행)
