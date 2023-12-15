import streamlit as st

# Load NLP Pkgs
import spacy
from spacy.lang.ko.examples import sentences 
from spacy import displacy
from textblob import TextBlob
import pandas as pd 
from collections import Counter
# Fxn to Get Wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import base64
import time

import plotly.express as px
# Text Cleaning Pkgs
import neattext as nt
import neattext.functions as nfx

# spaCy 라이브러리를 사용해 한국어 자연어 처리 모델을 로드하고, 이를 nlp라는 이름으로 참조
# spaCy 참고함 : https://spacy.io/models/ko
nlp = spacy.load("ko_core_news_lg")

# Raw 데이터 
def get_rawData(speaker_sentiment_sentence):
    # 빈 문자열을 초기화
    raw_text = ''
    # speaker_sentiment_sentence 딕셔너리를 순회. 이 딕셔너리의 키는 발화자(speaker), 값은 해당 발화자의 대화 내역(entries)입니다.
    for speaker, entries in speaker_sentiment_sentence.items():
        # 각 발화자의 대화 내역을 순회
        for entry in entries:
            # 대화 내역(entry)에서 텍스트를 추출해 raw_text에 추가
            raw_text += entry["text"]                

    # 모든 대화 내역을 하나로 합친 raw_text를 반환
    return raw_text


# Test
def text_analyzer_sample(rawtext):
    # spaCy의 nlp 함수를 사용하여 텍스트에 대한 자연어 처리를 수행
    doc = nlp(rawtext)

    # 수행한 텍스트를 출력
    st.write(doc.text)

    # 자연어 처리를 거친 텍스트(doc)의 각 토큰(단어)에 대해
    for token in doc:
        # 토큰의 텍스트, 품사(pos_), 의존성 구문 분석(dep_)을 출력
        st.write(token.text, token.pos_, token.dep_)

      
# 텍스트 분석 
def text_analyzer(rawtext):
	docx = nlp(rawtext)
	allData = [(token.text, token.shape_, token.pos_, token.tag_, token.lemma_, token.dep_, token.is_alpha, token.is_stop) for token in docx]
    # token.text: 현재 토큰의 텍스트 (문자열)
    # token.shape_: 현재 토큰의 모양 정보 (영어 단어의 경우 대소문자 패턴 등을 나타냄, 문자열)
    # token.pos_: 현재 토큰의 품사 정보 (Part of Speech, 문자열)
    # token.tag_: 현재 토큰의 세부적인 품사 태그 정보 (문자열)
    # token.lemma_: 현재 토큰의 기본형 (Lemma, 원형 단어, 문자열)
    # token.dep_: 현재 토큰의 의존성(dependency), 문법적인 관계, 주어(subject), 목적어(object), 보어(predicate nominative) 등이 의존성
    # token.is_alpha: 현재 토큰이 알파벳 문자인지 여부 (참(True) 또는 거짓(False), 불리언 값)
    # token.is_stop: 현재 토큰이 불용어(stopword)인지 여부 (참(True) 또는 거짓(False), 불리언 값)
    
	df = pd.DataFrame(allData, columns=['Token','Shape','PoS','Tag','Lemma','Dependency', 'IsAlpha','Is_Stopword'])
	return df 

def get_entities(my_text):
    # 주어진 텍스트를 spaCy의 nlp 객체에 넣어 언어 모델을 이용해 처리
    docx = nlp(my_text)

    # 처리된 텍스트(docx)에서 개체명(named entity)들을 추출
    # 각 개체명(entity)과 그 타입(label_)을 튜플 형태로 저장
    entities = [(entity.text, entity.label_) for entity in docx.ents]

    # 추출된 개체명들을 반환
    return entities


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
# @st.cache
def render_entities(rawtext):
    # 주어진 텍스트에 대해 spaCy의 nlp 객체를 이용해 자연어 처리
    docx = nlp(rawtext)

    # 개체명에 대한 색상 옵션 설정
    # PERSON (PS) - 개인 Individuals, including fictional ones.
    # NORP - 국가, 종교 또는 정치적 그룹 Nationalities, religious, or political groups.
    # FAC - 시설물 Buildings, airports, highways, bridges, etc.
    # ORG - 조직 Companies, agencies, institutions, etc.
    # GPE - 지리적 지역, 국가, 도시 Countries, cities, states.
    # LOC - 위치, 산, 강 등 Non-GPE locations, mountain ranges, bodies of water.
    # PRODUCT - 상품명 Objects, vehicles, foods, etc. (Not services.)
    # EVENT - 이벤트명 Named hurricanes, battles, wars, sports events, etc.
    # WORK_OF_ART - 예술 작품명 Titles of books, songs, etc.
    # LAW - 법률 문서명 Named documents made into laws.
    # LANGUAGE - 언어 Any named language.
    # DATE (DT) - 날짜 Absolute or relative dates or periods.
    # TIME (TI) - 시간 Times smaller than a day.
    # PERCENT - 퍼센트 값 Percentage, including "%".
    # MONEY - 화폐 금액 Monetary values, including unit.
    # QUANTITY - 수량 Measurements, as of weight or distance.
    # ORDINAL - 서수  "first", "second", etc.
    # CARDINAL - 기수 Numerals that do not fall under another type.
    
    # "PS": 사람, "QT": 인용구, "TI": 시간, "OG": 조직, "DT": 날짜, "LC": 위치에 대한 색상 설정
    options = {"colors": {"PS": "#f0ca78", "QT": "#82d980", "TI": "#9dbf8e", "OG": "#efcaf6", "DT": "#fcf9c7", "LC": "#afc3f9"}}

    # 처리된 텍스트(docx)에서 개체명을 시각화
    # displacy는 spaCy의 시각화 도구. "ent" 스타일로 개체명을 시각화하고 HTML로 출력
    html = displacy.render(docx, style="ent", page=True, options=options)
    # 출력된 HTML에서 불필요한 빈 줄을 제거
    html = html.replace("\n\n", "\n")
    # HTML을 보기 좋게 감싸는 외부 div에 적용
    result = HTML_WRAPPER.format(html)
    # 최종 결과를 반환
    return result


# 텍스트 데이터에서 가장 빈도가 높은 단어들을 추출하는 기능
def get_most_common_tokens(my_text, num=5):
    # 주어진 텍스트를 공백을 기준으로 나눠 토큰화한 후, 각 토큰의 등장 횟수를 세어 Counter 객체를 생성
    word_tokens = Counter(my_text.split())
    # Counter 객체의 most_common 메소드를 이용해 가장 많이 등장하는 토큰들을 추출
    # num 매개변수를 통해 추출할 토큰의 개수를 설정. num이 5라면 가장 많이 등장하는 5개의 토큰을 추출
    most_common_tokens = dict(word_tokens.most_common(num))
    # 가장 많이 등장하는 토큰들과 그 등장 횟수를 딕셔너리 형태로 반환
    return most_common_tokens


# Fxn to Get Sentiment
def get_sentiment(my_text):
    # TextBlob 객체를 생성. TextBlob는 자연어 처리를 위한 파이썬 라이브러리로, 감성 분석 등 다양한 기능을 제공
    blob = TextBlob(my_text)
    # TextBlob의 sentiment 메소드를 이용해 감성 분석을 수행. 
    # sentiment 메소드는 -1(매우 부정적)부터 1(매우 긍정적)까지의 값을 반환하는 polarity와 
    # 0(객관적)부터 1(주관적)까지의 값을 반환하는 subjectivity를 담은 NamedTuple을 반환
    sentiment = blob.sentiment
    # 감성 분석 결과를 반환
    return sentiment

#
def plot_word_freq(raw_text, num_of_most_common=5):
    # 불용어를 제거한 텍스트를 생성
    processed_text = nfx.remove_stopwords(raw_text)
    # 불용어가 제거된 텍스트에서 가장 많이 등장하는 단어들을 추출
    top_keywords = get_most_common_tokens(processed_text, num_of_most_common)
    # 가장 많이 등장하는 단어들을 바 차트로 시각화
    fig = px.bar(x=list(top_keywords.keys()), y=list(top_keywords.values()))
    # 차트의 타이틀과 x축, y축의 라벨을 설정하고, x축의 눈금 각도를 -45도로 설정
    fig.update_layout(title="Top Keywords/Frequency", xaxis_title="Keywords", yaxis_title="Frequency", xaxis_tickangle=-45)
    # 차트를 화면에 출력. streamlit의 plotly_chart() 함수를 사용
    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)


def plot_wordcloud(rawtext):
    # 사용할 폰트 파일의 경로를 설정. 여기서는 'NEXONLv1GothicBold'라는 폰트를 사용
    font_path = ".\\Fonts\\NEXONLv1GothicBold.ttf"  
    # WordCloud 객체를 생성.
    # font_path: 사용할 폰트의 경로
    # background_color: 워드클라우드의 배경색 ("white"로 설정)
    # width, height: 워드클라우드 이미지의 가로, 세로 크기
    # max_words: 워드클라우드에 표시할 최대 단어 수 (100으로 설정)
    # stopwords: 워드클라우드 생성 시 제외할 불용어 리스트 (STOPWORDS를 사용)
    # collocations: 같이 등장하는 단어를 한 그룹으로 묶을지의 여부 (False로 설정)
    wc = WordCloud(font_path=font_path, background_color="white", width=800, height=400,
                   max_words=100, stopwords=STOPWORDS, collocations=False)

    # 입력된 rawtext로 워드클라우드를 생성
    wordcloud = wc.generate(rawtext)
    # matplotlib의 figure 객체를 생성
    fig = plt.figure()
    # 워드클라우드 이미지를 화면에 출력. interpolation="bilinear"는 이미지 부드럽게 보이게 설정
    plt.imshow(wordcloud, interpolation="bilinear")
    # x축, y축의 눈금을 숨김
    plt.axis("off")
    # 화면에 그림을 출력. streamlit의 pyplot() 함수를 사용
    st.pyplot(fig)

#
def make_downloadable(data):
    # 현재 시간을 문자열로 변환. 포맷은 "YYYYMMDD-HHMMSS"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # 'data'를 CSV 형식으로 변환. 인덱스는 포함하지 않음.
    csvfile = data.to_csv(index=False)
    # CSV 파일을 인코딩하여 base64 문자열로 변환
    b64 = base64.b64encode(csvfile.encode()).decode()
    # 다운로드할 파일의 새 이름 생성
    new_filename = "nlp_result_{}_.csv".format(timestr)
    # 화면에 'Download CSV file'라는 문구를 출력
    st.write("### ** Download CSV file **")
    # CSV 파일을 다운로드할 수 있는 링크 생성. 링크는 base64로 인코딩된 CSV 데이터를 참조
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here!</a>'
    # 링크를 화면에 출력. HTML이 포함되어 있으므로 'unsafe_allow_html=True' 옵션 필요
    st.markdown(href, unsafe_allow_html=True)