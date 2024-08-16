import streamlit as st
import pdfplumber
import google.generativeai as genai
import os

def read_file(file_object):
    """파일 형식에 따라 파일 데이터를 읽어오는 함수"""
    try:
        if file_object.type == "application/pdf":
            pdf = pdfplumber.open(file_object)
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        elif file_object.type == "text/plain":
            return file_object.read().decode("utf-8")
        else:
            st.error("지원하지 않는 파일 형식입니다. (.pdf 또는 .txt 파일을 선택해주세요)")
            return None
    except Exception as e:
        st.error(f"파일을 읽는 동안 오류가 발생했습니다: {str(e)}")
        return None

def preprocess_specification(text, model, temperature=0.1):
    """Preprocesses specification text for comparison."""
    try:
        prompt = f"""Please remove any metadata from the following specification text, such as patent number, filing date, etc. 
        Then, segment the text into sentences, extract important keywords, and remove unnecessary sentences like background information to make it suitable for comparison.

        ## Specification Text:
        ```
        {text}
        ```

        ## Preprocessed Result:
        """
        generation_config = genai.GenerationConfig(temperature=temperature)
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        st.error(f"Error occurred while preprocessing specification text with AI: {str(e)}")
        return None

def preprocess_claims(text, model, temperature=0.1):
    """Preprocesses claims text for comparison."""
    try:
        prompt = f"""For the following claims text, please extract each claim number. 
        Then, segment each claim into sentences and extract important keywords to make it suitable for comparison.

        ## Claims Text:
        ```
        {text}
        ```

        ## Preprocessed Result:
        """
        generation_config = genai.GenerationConfig(temperature=temperature)
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        st.error(f"Error occurred while preprocessing claims text with AI: {str(e)}")
        return None

def compare_texts(text1, text2, model, temperature=0.1):
    """Compares two texts using AI and provides results in Korean."""
    try:
        prompt = f"""
        Compare the following texts and evaluate the similarity.
        Indicate if each claim from Text 2 is included in Text 1.
        Consider similar expressions/words as identical.
        Text 1 (Prior application specification):
        {text1}

        Text 2 (Later application claims):
        {text2}

        Result format(only display the claims from the text 2 in the table) :
        | 청구항 번호 | Included? | Similarity | Reasoning |
        |--------------|-----------|------------|-----------|
        | ...          | ...       | ...        | ...       |

        Similarity Scale:
        - Very High (90-100%): Almost identical content
        - High (70-89%): Most key elements match
        - Medium (50-69%): Some key elements match
        - Low (30-49%): Few elements match
        - Very Low (0-29%): Almost no match

        For each claim, briefly explain the reasoning behind your similarity judgment.
        After completing the table, summarize the overall similarity analysis results.

        Please provide all results in Korean.
        """
        generation_config = genai.GenerationConfig(temperature=temperature)
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        st.error(f"AI를 사용한 텍스트 비교 중 오류 발생: {str(e)}")
        return None


def get_api_key():
    """사용자로부터 API 키 앞 2글자와 뒤 4글자를 입력받아 완성하는 함수"""
    api_key_middle = "zaSyBjhTX0EWpHXdvpYm9Dhk-fZFWLyU_"  # API 키 중간 부분
    user_input = st.text_input("API 키를 입력하세요:", type="password")
    if len(user_input) != len(api_key_middle) + 6:
        st.error("API 키를 정확하게 입력해야 합니다.")
        return None
    api_key_prefix = user_input[:2]  # 입력값에서 앞 2글자 추출
    api_key_suffix = user_input[-4:]  # 입력값에서 뒤 4글자 추출
    return api_key_prefix + api_key_middle + api_key_suffix

def main():
    """Streamlit 웹 애플리케이션의 메인 함수"""
    st.title("문서 비교 도구")

    # 사용 방법 안내
    st.markdown("""
    ## 사용 방법:
    1. 비교할 두 개의 파일을 업로드하거나 텍스트를 직접 입력합니다.
    2. "비교 시작" 버튼을 클릭합니다.
    3. 결과 창에서 두 문서의 유사도 분석 결과를 확인합니다.

    ## 주의 사항:
    * **초기 버전**: 현재 초기 버전이므로 오류가 발생할 수 있습니다.
    * **이미지 인식 불가**: 화학식, 도면 등의 이미지는 인식하지 못할 수 있습니다. 텍스트 변환 후 사용하세요.
    * **파일 크기 제한**: 외부 AI API를 이용하므로, 일정 용량 이상의 파일은 인식이 어려울 수 있습니다.
    * **PDF 형식**: 스캔된 PDF 파일은 텍스트 추출이 제대로 되지 않을 수 있습니다. 
    * **정확도**: AI 기반 분석 결과는 참고용이며, 법적/전문적인 판단을 대신할 수 없습니다. 
    """)

    # API 키 입력
    api_key = get_api_key()
    if api_key is None:
        return

    # 파일 업로드
    prior_file = st.file_uploader("비교 대상 명세서 (텍스트 1) 파일 업로드 (.pdf 또는 .txt)", type=['pdf', 'txt'])
    later_file = st.file_uploader("비교 대상 청구항 (텍스트 2) 파일 업로드 (.pdf 또는 .txt)", type=['pdf', 'txt'])

    # 텍스트 직접 입력
    prior_text_input = st.text_area("비교 대상 명세서 (텍스트 1) 직접 입력", height=10)
    later_text_input = st.text_area("비교 대상 청구항 (텍스트 2) 직접 입력", height=10)

    # 비교 실행 버튼
    if st.button("비교 시작"):
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro-001')

            # 파일 또는 텍스트 입력 선택
            if prior_file and later_file:
                prior_text = read_file(prior_file)
                later_text = read_file(later_file)
            elif prior_text_input and later_text_input:
                prior_text = prior_text_input
                later_text = later_text_input
            else:
                st.error("두 텍스트를 모두 입력하거나 파일을 업로드하세요.")
                return

            if prior_text is None or later_text is None:
                return

            # 텍스트 전처리
            with st.spinner("텍스트 전처리 중..."):
                processed_prior_text = preprocess_specification(prior_text, model)
                processed_later_text = preprocess_claims(later_text, model)

            if processed_prior_text is None or processed_later_text is None:
                return

            # 비교 수행
            with st.spinner("비교 중..."):
                comparison_result = compare_texts(processed_prior_text, processed_later_text, model)

            if comparison_result is None:
                return

            st.subheader("비교 결과:")
            st.text(comparison_result)

        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
