@echo off
echo Running the Python script in the VENV environment...

REM 가상 환경 경로 설정, 가상 환경이 있는 디렉터리를 확인하세요.
set VENV_PATH=.venvRyan\Scripts\activate
set REQUIREMENTS_PATH=requirements.txt

REM 가상 환경이 존재하는지 확인
if not exist %VENV_PATH% (
    echo Virtual environment not found. Creating new one...
    python -m venv .venvRyan
    echo Entering the virtual environment...
    call %VENV_PATH%
    echo Installing requirements from %REQUIREMENTS_PATH%
    if exist %REQUIREMENTS_PATH% (
        pip install -r %REQUIREMENTS_PATH%
    ) else (
        echo Requirements file not found. Check the path: %REQUIREMENTS_PATH%
    )
) else (
    echo Entering the virtual environment...
    call %VENV_PATH%
)

set START_TIME=%time%

REM 화면실행 
streamlit run app.py --server.port=8080

REM Streamlit이 먼저 실행된 후에 웹 브라우저를 열도록, timeout 추가
timeout /t 5 /nobreak > NUL
start chrome.exe http://localhost:8080

set END_TIME=%time%

REM 가상 환경 비활성화
call .venvRyan\Scripts\deactivate

@echo on
echo Started at: %START_TIME%
echo Finished at: %END_TIME%

REM pause