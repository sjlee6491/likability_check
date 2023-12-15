'''
import streamlit as st
import streamlit.components.v1 as stc

#메인로직 및 공통함수점검 
import app_common as app_common

#필요한 모듈
import numpy as np
import os

#keras
import tensorflow as tf
import tensorflow.keras as krs

MODEL_TRAINED_PATH = './model/Trained'
MODEL_TAGNAME = '_Trained'

def page_main():
    st.subheader("분석작업 #5 : 모델기반 학습하기")  
    app_common.set_side_fileinfo()
    st.info("처리내용: 채팅데이터 학습모델만들기 ")
    
    #st.write("test 1")
    # 클린징 값이 들어왔으면 아래 로직 수행 (클린징은 1_Data_Cleansing.py 로직에서 수행)
    if "speaker_sentiment_sentence" in st.session_state:
        if st.session_state.speaker_sentiment_sentence is not None:
            speaker_sentiment_sentence = st.session_state.speaker_sentiment_sentence
            
            # 채팅 메세지에 speakers 인원별로 동적생성             
            keys = speaker_sentiment_sentence.keys()
            num_of_speakers = len(keys)
            cols = st.columns(num_of_speakers)
            keys = list(keys)
            for i in range(num_of_speakers):
                key = keys[i]
                key_no_space = key.replace(" ", "")  # 띄어쓰기 제거
                key_no_space = key_no_space.encode("utf-8").decode("ascii", "ignore")  # 한국어 문자 제거
                entries = speaker_sentiment_sentence[key]
                # 'entries'에서 'text'만 추출하여 'texts' 리스트에 넣기
                texts = [entry["text"] for entry in entries]
                with cols[i]:
                    # 비교모델의 번호를 동적으로 할당
                    st.markdown(f"- 비교모델{i+1} : <span style='color:blue'>{key}</span>", unsafe_allow_html=True) 
                    set_learning(texts, f"Model_{key_no_space}v{i+1}")  # 모델 이름도 동적으로 할당

                #st.write(texts)
                # col1, col2, col3 = st.columns([2, 4, 4])
                # with col1: 
                #     st.write("- 분류 DataSet")
                #     sInfo  = f"- 학습데이터 (Text): {len(texts)}" + "\n" 
                #     st.caption(sInfo)        
                #     st.write(texts)

# 문장 생성
def generate_sentence(model, tokenizer, current_word, n, max_len):
    init_word = current_word
    sentence = ''

    for _ in range(n):
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = krs.preprocessing.sequence.pad_sequences([encoded], maxlen=max_len, padding='pre')
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        for word, index in tokenizer.word_index.items():
            if index == result:
                break

        current_word = current_word + ' ' + word
        sentence = sentence + ' ' + word

    sentence = init_word + sentence
    return sentence

#모델학습
def set_learning(texts, pkeyid="Model1"):
    keyid = pkeyid
    # 텍스트를 단어로 토큰화하여 단어 집합 생성
    tokenizer = krs.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1
    
    sInfo  = f"- 검토문장수(document_count): {tokenizer.document_count}" + "\n" 
    sInfo += f"- 토큰단어수(word_index): {vocab_size}" + "\n" 
    sInfo += f"- 필터조건(filters): {tokenizer.filters}" + "\n" 
    st.caption(sInfo)
        
    #모델이 존재하는지 여부 확인     
    filelists, bExist = exist_model(keyid)
    s_checkmodel = "- 모델존재여부확인" + "\n"
    for key, value in filelists.items():
        s_checkmodel += f"- Model: {key} / existed : {value}" + "\n"
    st.caption(s_checkmodel)
    
    with st.expander("Model Selector : " + keyid):
        bProcess = True
        sModel = st.radio("모델을 선택하세요", (f'{keyid}모델학습', f'{keyid}모델사용'), horizontal=True, index=1, key=f"{keyid}_radio")
        if sModel == f'{keyid}모델학습':
            with st.form(key=f"{keyid}_form"):
                embedding_dim = st.number_input('임베딩 차원 수 입력', min_value=2, value=10, max_value=100, key=f"{keyid}_embed")
                hidden_units = st.number_input('은닉층의 뉴런 수 입력', min_value=2, value=32, max_value=100, key=f"{keyid}_hidden")
                epochs = st.number_input('에포크 수 입력', min_value=100, value=200, max_value=1000, key=f"{keyid}_epochs")
                submit_button = st.form_submit_button(label=f"{keyid}_모델생성하기")
            
            if submit_button:
                with st.spinner(f"Model {keyid} is training..."):
                    X, y, vocab_size, tokenizer, max_len  = preprocess_data(tokenizer, texts)                
                    model = train_model(X, y, vocab_size, embedding_dim, hidden_units, epochs, keyid)
                    save_model(model, tokenizer, max_len, keyid)
            else :
                bProcess = False    
        elif sModel == f'{keyid}모델사용':
            if bExist :
                model, tokenizer, max_len = load_model(keyid)  
            else :
                st.error(f"Model {keyid}이 없습니다. 모댈 재학습을 진행하세요")
                bProcess = False

        # bProcess 변수가 True일 때 아래 코드를 실행
        if bProcess :
            answer = st.text_input("예측할 말을 입력하세요", "", key=f"{keyid}_text_input")
            
            # 사용자에게 슬라이더를 통해 예측할 단어의 수를 입력받음 (최소 1개, 최대 max_len개, 초기값은 max_len의 절반)
            times = st.slider("예측할 단어의 수", min_value=1, max_value=max_len, value=int(max_len/2), key=f"{keyid}_slider")
            if answer != "" :  # 사용자가 입력한 문장(answer)       
                m_data = generate_sentence(model, tokenizer, answer, int(times), max_len) #입력 문장에 이어질 문장을 생성
                st.success(m_data)
            pass
            

# 데이터전처리
def preprocess_data(tokenizer, texts):
    sequences = list()
    for line in texts:
        
        encoded = tokenizer.texts_to_sequences([line])[0]
        #st.write(encoded)        
        for i in range(1, len(encoded)):
            sequence = encoded[:i + 1]
            sequences.append(sequence)
    l=texts[0]
    a=0
    for line in texts:
        
        encoded = tokenizer.texts_to_sequences([line])[0]
        sequences.append(tokenizer.texts_to_sequences([l]))
        #st.write(encoded)        
        for i in range(1, len(encoded)):
            sequence = encoded[:i + 1]
            sequences[a]=sequences[a]+sequence
        a+=1
        l=line
    print(sequences)
    # 모든 샘플에서 가장 긴 샘플의 길이로 패딩
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max(len(l) for l in sequences)
    sequences = krs.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding='pre')
    sequences = np.array(sequences)
    
    st.caption(f"- 최대단어길이(max_len): {max_len}")
    st.write(sequences)
    X = sequences[:, :-1]
    Y = sequences[:, -1]
    y = krs.utils.to_categorical(Y, num_classes=vocab_size) # 레이블을 원-핫 인코딩
    #st.write(f"X: {X}, Y: {Y}, y: {y}") #확인용

    return X, y, vocab_size, tokenizer, max_len

# keras의 Callback 클래스를 상속받아 CustomCallback 클래스를 정의
class CustomCallback(krs.callbacks.Callback):
    def __init__(self, container, epochs, keyid): # 생성자에 container 인자 추가
        super().__init__()          # 상위 클래스의 생성자 호출
        self.container = container  # container 속성 설정
        self.epochs = epochs        # epochs 변수
        self.keyid = keyid          # keyid 변수로 저장
    
    # 에포크가 끝날 때마다 호출되는 함수를 재정의 (override)
    def on_epoch_end(self, epoch, logs=None):
        # logs 인자가 None이 아니라면 (즉, 학습 로그가 존재하면)
        if logs is not None:
            with self.container.container():   # self.container를 이용하여 에포크 번호와 학습 로그를 출력
                st.caption(f"- Model {self.keyid}, 학습중  Epoch {epoch + 1} / {self.epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}")

# 모델 학습
def train_model(X, y, vocab_size, pEmbedding_dim=10, pHidden_units=32, pEpochs=200, pkeyid="Model1"):
    keyid = pkeyid
    
    #임베딩 차원의 수, 자연어처리에서 각 단어를 고정된 길이의 밀집 벡터로 변환, 밀집 벡터의 길이, 즉 벡터가 가지는 특성의 수를 결정
    #임베딩 차원이 높을수록 단어 사이의 복잡한 관계를 더 잘 모델링 할 수 있지만, 더 많은 계산량과 데이터가 필요함
    embedding_dim = 10 
    
    #순환 신경망(RNN)의 은닉층의 뉴런 수
    #신경망의 용량을 결정하며, 은닉층의 뉴런이 많을수록 복잡한 패턴을 학습
    #더 많은 계산량과 데이터가 필요하고 과적합의 위험이 증가
    hidden_units = pHidden_units

    #신경망을 학습시키는 데 사용되는 전체 데이터 세트에 대한 훈련 반복 횟수
    #많은 에포크 수로 학습을 실행하면 모델이 데이터에서 더 많은 패턴을 학습할 수 있지만, 과적합의 위험이 증가
    epochs = pEpochs

    #tensorflow.keras.models
    model = krs.models.Sequential()
    model.add(krs.layers.Embedding(vocab_size, embedding_dim))
    model.add(krs.layers.SimpleRNN(hidden_units))
    model.add(krs.layers.Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # CustomCallback 객체 생성 : 모델 학습시에 콜백 추가
    container = st.empty()  # 빈 컨테이너 생성 
    custom_callback = CustomCallback(container, epochs, keyid)
    
    # 모델 학습시에 콜백 추가
    model.fit(X, y, epochs=epochs, verbose=2, callbacks=[custom_callback])

    return model

# 모델 저장 함수 정의
def save_model(model, tokenizer, max_len, pkeyid="Model1"):
    keyid = pkeyid
    sFileKey = MODEL_TRAINED_PATH + "/" + keyid + MODEL_TAGNAME
    model.save(sFileKey + ".keras")  # 모델 저장
    tokenizer_json = tokenizer.to_json()  # 토크나이저 정보 저장

    # 토크나이저 정보 저장
    with open(sFileKey + "_tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer_json)

    # 최대 길이 정보 저장
    with open(sFileKey + "_maxlen.txt", "w", encoding="utf-8") as f:
        f.write(str(max_len))  

# 모델파일존재여부 확인
def exist_model(pkeyid="Model1"): 
    keyid = pkeyid   
    if not os.path.exists(MODEL_TRAINED_PATH):
        os.makedirs(MODEL_TRAINED_PATH)

    sFileKey = MODEL_TRAINED_PATH + "/" + keyid + MODEL_TAGNAME    
    ext_names = [".keras", "_tokenizer.json", "_maxlen.txt"] # 3개의 파일 이름 리스트
    
    file_existence_list = {} # 파일 존재 여부를 담을 리스트 초기화, 각 파일 이름을 검사하여 존재 여부를 리스트에 추가
    bFileExist = True
    for ext_name in ext_names:
        file_existence = os.path.exists(sFileKey + ext_name)
        file_existence_list[keyid + MODEL_TAGNAME + ext_name] = file_existence
        if bFileExist != file_existence:
            bFileExist = False 

    return file_existence_list, bFileExist

# 모델 불러오기 함수 정의
def load_model(pkeyid="Model1"):
    keyid = pkeyid
    sFileKey = MODEL_TRAINED_PATH + "/" + keyid + MODEL_TAGNAME

    st.write(sFileKey)
    model = krs.models.load_model(f"{sFileKey}.keras")  # 저장된 모델 불러오기
    with open(f"{sFileKey}_tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
        tokenizer = krs.preprocessing.text.tokenizer_from_json(tokenizer_json)  # 저장된 토크나이저 불러오기

    with open(f"{sFileKey}_maxlen.txt", "r", encoding="utf-8") as f:
        max_len = int(f.read())  # 저장된 최대 길이 불러오기

    return model, tokenizer, max_len


#최초실행시 호출 
if __name__ == "__main__":
    if app_common.session_check():
        page_main()
    else :
        st.title("메인페이지로 돌아가세요")
'''
import streamlit as st
import streamlit.components.v1 as stc

#메인로직 및 공통함수점검 
import app_common as app_common

#필요한 모듈
import numpy as np
import os

#keras
import tensorflow as tf
import tensorflow.keras as krs

MODEL_TRAINED_PATH = './model/Trained'
MODEL_TAGNAME = '_Trained'

def page_main():
    st.subheader("분석작업 #5 : 모델기반 학습하기")  
    app_common.set_side_fileinfo()
    st.info("처리내용: 채팅데이터 학습모델만들기 ")
    
    #st.write("test 1")
    # 클린징 값이 들어왔으면 아래 로직 수행 (클린징은 1_Data_Cleansing.py 로직에서 수행)
    if "speaker_sentiment_sentence" in st.session_state:
        if st.session_state.speaker_sentiment_sentence is not None:
            speaker_sentiment_sentence = st.session_state.speaker_sentiment_sentence
            
            # 채팅 메세지에 speakers 인원별로 동적생성             
            keys = speaker_sentiment_sentence.keys()
            num_of_speakers = len(keys)
            cols = st.columns(num_of_speakers)
            keys = list(keys)
            for i in range(num_of_speakers):
                key = keys[i]
                key_no_space = key.replace(" ", "")  # 띄어쓰기 제거
                key_no_space = key_no_space.encode("utf-8").decode("ascii", "ignore")  # 한국어 문자 제거
                entries = speaker_sentiment_sentence[key]
                # 'entries'에서 'text'만 추출하여 'texts' 리스트에 넣기
                texts = [entry["text"] for entry in entries]
                with cols[i]:
                    # 비교모델의 번호를 동적으로 할당
                    st.markdown(f"- 비교모델{i+1} : <span style='color:blue'>{key}</span>", unsafe_allow_html=True) 
                    set_learning(texts, f"Model_{key_no_space}v{i+1}")  # 모델 이름도 동적으로 할당

                #st.write(texts)
                # col1, col2, col3 = st.columns([2, 4, 4])
                # with col1: 
                #     st.write("- 분류 DataSet")
                #     sInfo  = f"- 학습데이터 (Text): {len(texts)}" + "\n" 
                #     st.caption(sInfo)        
                #     st.write(texts)

# 문장 생성
def generate_sentence(model, tokenizer, current_word, n, max_len):
    init_word = current_word
    sentence = ''

    for _ in range(n):
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = krs.preprocessing.sequence.pad_sequences([encoded], maxlen=max_len, padding='pre')
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        for word, index in tokenizer.word_index.items():
            if index == result:
                break

        current_word = current_word + ' ' + word
        sentence = sentence + ' ' + word

    sentence = init_word + sentence
    return sentence

#모델학습
def set_learning(texts, pkeyid="Model1"):
    keyid = pkeyid
    # 텍스트를 단어로 토큰화하여 단어 집합 생성
    tokenizer = krs.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1
    
    sInfo  = f"- 검토문장수(document_count): {tokenizer.document_count}" + "\n" 
    sInfo += f"- 토큰단어수(word_index): {vocab_size}" + "\n" 
    sInfo += f"- 필터조건(filters): {tokenizer.filters}" + "\n" 
    st.caption(sInfo)
        
    #모델이 존재하는지 여부 확인     
    filelists, bExist = exist_model(keyid)
    s_checkmodel = "- 모델존재여부확인" + "\n"
    for key, value in filelists.items():
        s_checkmodel += f"- Model: {key} / existed : {value}" + "\n"
    st.caption(s_checkmodel)
    
    with st.expander("Model Selector : " + keyid):
        bProcess = True
        sModel = st.radio("모델을 선택하세요", (f'{keyid}모델학습', f'{keyid}모델사용'), horizontal=True, index=1, key=f"{keyid}_radio")
        if sModel == f'{keyid}모델학습':
            with st.form(key=f"{keyid}_form"):
                embedding_dim = st.number_input('임베딩 차원 수 입력', min_value=2, value=10, max_value=100, key=f"{keyid}_embed")
                hidden_units = st.number_input('은닉층의 뉴런 수 입력', min_value=2, value=32, max_value=100, key=f"{keyid}_hidden")
                epochs = st.number_input('에포크 수 입력', min_value=100, value=200, max_value=1000, key=f"{keyid}_epochs")
                submit_button = st.form_submit_button(label=f"{keyid}_모델생성하기")
            
            if submit_button:
                with st.spinner(f"Model {keyid} is training..."):
                    X, y, vocab_size, tokenizer, max_len  = preprocess_data(tokenizer, texts)                
                    model = train_model(X, y, vocab_size, embedding_dim, hidden_units, epochs, keyid)
                    save_model(model, tokenizer, max_len, keyid)
            else :
                bProcess = False    
        elif sModel == f'{keyid}모델사용':
            if bExist :
                model, tokenizer, max_len = load_model(keyid)  
            else :
                st.error(f"Model {keyid}이 없습니다. 모델 재학습을 진행하세요")
                bProcess = False

        # bProcess 변수가 True일 때 아래 코드를 실행
        if bProcess :
            answer = st.text_input("예측할 말을 입력하세요", "", key=f"{keyid}_text_input")
            
            # 사용자에게 슬라이더를 통해 예측할 단어의 수를 입력받음 (최소 1개, 최대 max_len개, 초기값은 max_len의 절반)
            times = st.slider("예측할 단어의 수", min_value=1, max_value=max_len, value=int(max_len/2), key=f"{keyid}_slider")
            if answer != "" :  # 사용자가 입력한 문장(answer)       
                m_data = generate_sentence(model, tokenizer, answer, int(times), max_len) #입력 문장에 이어질 문장을 생성
                st.success(m_data)
            pass
            

# 데이터전처리
def preprocess_data(tokenizer, texts):
    sequences = list()
    for line in texts:
        encoded = tokenizer.texts_to_sequences([line])[0]
        #st.write(encoded)        
        for i in range(1, len(encoded)):
            sequence = encoded[:i + 1]
            sequences.append(sequence)
        
    # 모든 샘플에서 가장 긴 샘플의 길이로 패딩
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max(len(l) for l in sequences)
    sequences = krs.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding='pre')
    sequences = np.array(sequences)
    
    st.caption(f"- 최대단어길이(max_len): {max_len}")
    st.write(sequences)
    X = sequences[:, :-1]
    Y = sequences[:, -1]
    y = krs.utils.to_categorical(Y, num_classes=vocab_size) # 레이블을 원-핫 인코딩
    #st.write(f"X: {X}, Y: {Y}, y: {y}") #확인용

    return X, y, vocab_size, tokenizer, max_len

# keras의 Callback 클래스를 상속받아 CustomCallback 클래스를 정의
class CustomCallback(krs.callbacks.Callback):
    def __init__(self, container, epochs, keyid): # 생성자에 container 인자 추가
        super().__init__()          # 상위 클래스의 생성자 호출
        self.container = container  # container 속성 설정
        self.epochs = epochs        # epochs 변수
        self.keyid = keyid          # keyid 변수로 저장
    
    # 에포크가 끝날 때마다 호출되는 함수를 재정의 (override)
    def on_epoch_end(self, epoch, logs=None):
        # logs 인자가 None이 아니라면 (즉, 학습 로그가 존재하면)
        if logs is not None:
            with self.container.container():   # self.container를 이용하여 에포크 번호와 학습 로그를 출력
                st.caption(f"- Model {self.keyid}, 학습중  Epoch {epoch + 1} / {self.epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}")

# 모델 학습
def train_model(X, y, vocab_size, pEmbedding_dim=10, pHidden_units=32, pEpochs=200, pkeyid="Model1"):
    keyid = pkeyid
    
    #임베딩 차원의 수, 자연어처리에서 각 단어를 고정된 길이의 밀집 벡터로 변환, 밀집 벡터의 길이, 즉 벡터가 가지는 특성의 수를 결정
    #임베딩 차원이 높을수록 단어 사이의 복잡한 관계를 더 잘 모델링 할 수 있지만, 더 많은 계산량과 데이터가 필요함
    embedding_dim = 10 
    
    #순환 신경망(RNN)의 은닉층의 뉴런 수
    #신경망의 용량을 결정하며, 은닉층의 뉴런이 많을수록 복잡한 패턴을 학습
    #더 많은 계산량과 데이터가 필요하고 과적합의 위험이 증가
    hidden_units = pHidden_units

    #신경망을 학습시키는 데 사용되는 전체 데이터 세트에 대한 훈련 반복 횟수
    #많은 에포크 수로 학습을 실행하면 모델이 데이터에서 더 많은 패턴을 학습할 수 있지만, 과적합의 위험이 증가
    epochs = pEpochs

    #tensorflow.keras.models
    model = krs.models.Sequential()
    model.add(krs.layers.Embedding(vocab_size, embedding_dim))
    model.add(krs.layers.SimpleRNN(hidden_units))
    model.add(krs.layers.Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # CustomCallback 객체 생성 : 모델 학습시에 콜백 추가
    container = st.empty()  # 빈 컨테이너 생성 
    custom_callback = CustomCallback(container, epochs, keyid)
    
    # 모델 학습시에 콜백 추가
    model.fit(X, y, epochs=epochs, verbose=2, callbacks=[custom_callback])

    return model

# 모델 저장 함수 정의
def save_model(model, tokenizer, max_len, pkeyid="Model1"):
    keyid = pkeyid
    sFileKey = MODEL_TRAINED_PATH + "/" + keyid + MODEL_TAGNAME
    model.save(sFileKey + ".keras")  # 모델 저장
    tokenizer_json = tokenizer.to_json()  # 토크나이저 정보 저장

    # 토크나이저 정보 저장
    with open(sFileKey + "_tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer_json)

    # 최대 길이 정보 저장
    with open(sFileKey + "_maxlen.txt", "w", encoding="utf-8") as f:
        f.write(str(max_len))  

# 모델파일존재여부 확인
def exist_model(pkeyid="Model1"): 
    keyid = pkeyid   
    if not os.path.exists(MODEL_TRAINED_PATH):
        os.makedirs(MODEL_TRAINED_PATH)

    sFileKey = MODEL_TRAINED_PATH + "/" + keyid + MODEL_TAGNAME    
    ext_names = [".keras", "_tokenizer.json", "_maxlen.txt"] # 3개의 파일 이름 리스트
    
    file_existence_list = {} # 파일 존재 여부를 담을 리스트 초기화, 각 파일 이름을 검사하여 존재 여부를 리스트에 추가
    bFileExist = True
    for ext_name in ext_names:
        file_existence = os.path.exists(sFileKey + ext_name)
        file_existence_list[keyid + MODEL_TAGNAME + ext_name] = file_existence
        if bFileExist != file_existence:
            bFileExist = False 

    return file_existence_list, bFileExist

# 모델 불러오기 함수 정의
def load_model(pkeyid="Model1"):
    keyid = pkeyid
    sFileKey = MODEL_TRAINED_PATH + "/" + keyid + MODEL_TAGNAME

    st.write(sFileKey)
    model = krs.models.load_model(f"{sFileKey}.keras")  # 저장된 모델 불러오기
    with open(f"{sFileKey}_tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
        tokenizer = krs.preprocessing.text.tokenizer_from_json(tokenizer_json)  # 저장된 토크나이저 불러오기

    with open(f"{sFileKey}_maxlen.txt", "r", encoding="utf-8") as f:
        max_len = int(f.read())  # 저장된 최대 길이 불러오기

    return model, tokenizer, max_len


#최초실행시 호출 
if __name__ == "__main__":
    if app_common.session_check():
        page_main()
    else :
        st.title("메인페이지로 돌아가세요")