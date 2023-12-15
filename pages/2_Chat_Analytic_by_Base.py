import streamlit as st
import streamlit.components.v1 as stc

#메인로직 및 공통함수점검 
import app_common as app_common
from app_utils import *


def page_main(): 
    st.subheader("분석작업 #2 : 데이터 언어정보 기본 분석")  
    app_common.set_side_fileinfo()
    st.info("처리내용: 데이터 정보 언어적 특성확인 ")

    # 클린징 값이 들어왔으면 아래 로직 수행 (클린징은 1_Data_Cleansing.py 로직에서 수행)
    if "speaker_sentiment_sentence" in st.session_state:
        if st.session_state.speaker_sentiment_sentence is not None:
            speaker_sentiment_sentence = st.session_state.speaker_sentiment_sentence
           
            st.sidebar.caption("Option")
            num_of_most_common = st.sidebar.number_input("Most Common Tokens", 7, 15)

            if st.button("Analyze/분석하기"):
                #raw_text = get_rawData(speaker_sentiment_sentence)
                raw_text = st.session_state.chatmsg_fulltext
                with st.expander("Original Text"):
                    st.write(raw_text)

                with st.expander("Text Analysis"):
                    token_result_df = text_analyzer(raw_text)
                    st.dataframe(token_result_df)

                with st.expander("Entities"):
                    # entity_result = get_entities(raw_text)
                    # st.write(entity_result)

                    entity_result = render_entities(raw_text)
                    stc.html(entity_result, height=1000, scrolling=True)    

                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("Word Stats 텍스트 길이, 모음, 자음, 불용어, 모음빈도, 자음빈도 확인"):
                        st.info("Word Statistics")
                        docx = nt.TextFrame(raw_text)
                        st.write(docx.word_stats())
                        # "Length of Text": 텍스트의 총 길이를 의미합니다. 이 값은 59750입니다.
                        # "Num of Vowels": 텍스트에 포함된 모음 (vowel)의 개수를 의미합니다. 이 값은 339입니다.
                        # "Num of Consonants": 텍스트에 포함된 자음 (consonant)의 개수를 의미합니다. 이 값은 795입니다.
                        # "Num of Stopwords": 텍스트에 포함된 불용어 (stopwords)의 개수를 의미합니다. 불용어는 자연어 처리 작업에서 제외되는 일반적인 단어들을 말합니다. 이 값은 8입니다.
                        # "Stats of Vowels": 모음들의 빈도를 나타내는 사전(dictionary)입니다. 각 모음 (a, e, i, o, u)에 해당하는 횟수를 보여줍니다. 예를 들어, "a": 73은 텍스트에서 'a'라는 모음이 총 73번 나타난다는 것을 의미합니다.
                        # "Stats of Consonants": 자음들의 빈도를 나타내는 사전(dictionary)입니다. 각 자음에 해당하는 횟수를 보여줍니다. 예를 들어, "b": 25는 'b'라는 자음이 총 25번 나타난다는 것을 의미합니다.
                    
                    with st.expander("Top Keywords 가장 빈도가 높은 단어들"):
                        st.info("Top Keywords/Tokens")
                        #함수를 사용하여 텍스트에서 불용어(Stopwords)를 제거후 processed_text 넣음 
                        processed_text = nfx.remove_stopwords(raw_text)
                        keywords = get_most_common_tokens(
                            processed_text, num_of_most_common
                        )
                        st.write(keywords)
                    with st.expander("Sentiment 긍정성(polarity):-1(부정)~1(긍정), 주관성(subjectivity: 0(객관)~1(주관))"):
                        sent_result = get_sentiment(raw_text)
                        st.write(sent_result)
                with col2:
                    with st.expander("Plot Word Freq"):
                        plot_word_freq(raw_text, num_of_most_common)
                        # fig = plt.figure()
                        # top_keywords = get_most_common_tokens(
                        #     processed_text, num_of_most_common
                        # )
                        # plt.bar(keywords.keys(), top_keywords.values())
                        # plt.xticks(rotation=45)
                        # st.pyplot(fig)
                    with st.expander("Plot Part of Speech"):
                        try:
                            fig = plt.figure()
                            sns.countplot(token_result_df["PoS"])
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                        except:
                            st.warning("Insufficient Data: Must be more than 2")

                    with st.expander("Plot Wordcloud"):
                        try:
                            plot_wordcloud(raw_text)
                        except:
                            st.warning("Insufficient Data: Must be more than 2")
            # with st.expander("Download 분석결과 다운로드"):
            #     st.write(token_result_df)
            #     make_downloadable(token_result_df)
            #df = get_dataframe(speaker_sentiment_sentence) 
            #st.write(df)


# 분석데이터를 데이터프레임에 넣기 
def get_dataframe(speaker_sentiment_sentence):
    speakers = []
    dates = []
    timestamps = []
    texts = []
   
    for speaker, entries in speaker_sentiment_sentence.items():
        for entry in entries:
            speakers.append(speaker)
            dates.append(entry["date"])
            timestamps.append(entry["timestamp"])
            texts.append(entry["text"])      

    # 리스트들을 딕셔너리로 변환하여 데이터프레임 생성
    df_data = {
        "Speaker": speakers,
        "Date": dates,
        "Timestamp": timestamps,
        "Text": texts
    }

    df = pd.DataFrame(df_data)
    return df


#최초실행시 호출 
if __name__ == "__main__":
    if app_common.session_check():
        page_main()
    else :
        st.title("메인페이지로 돌아가세요")