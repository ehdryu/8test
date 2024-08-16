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

def process_text_with_gemini(text, model, instructions=""):
    """AI를 사용하여 텍스트를 비교에 적합하게 전처리합니다.
    추가적인 지시 사항을 instructions에 입력할 수 있습니다.
    """
    try:
        prompt = f"{instructions}\n\nPlease preprocess the following text into a suitable format for comparison:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"AI를 사용한 텍스트 처리 중 오류 발생, 비밀번호가 정확한지 확인하고, 문서 양이 너무 큰지 확인해주세요: {str(e)}")
        return None

def compare_texts(text1, text2, model):
    """두 텍스트의 유사도를 AI를 이용하여 비교하고 결과를 한글로 제공합니다."""
    try:
        prompt = f"""
        Compare the following texts and evaluate the similarity.
        Indicate if each claim from Text 2 is included in Text 1.
        Consider similar expressions/words as identical.
        Text 1 (Prior application specification):
        {text1}

        Text 2 (Later application claims):
        {text2}

        Result format:
        Please provide the results in the following table format (Please only display the claims from the text 2 in the table) :
        | 청구항 번호 | Included? | Similarity | Reasoning |
        |--------------|-----------|------------|-----------|
        | 청구항 1          | ...       | ...        | ...       |
        | 청구항 2          | ...       | ...        | ...       |
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
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"AI를 사용한 텍스트 처리 중 오류 발생, 비밀번호가 정확한지 확인하고, 문서 양이 너무 큰지 확인해주세요: {str(e)}")
        return None


def get_api_key():
    """사용자로부터 API 키 앞 2글자와 뒤 4글자를 입력받아 완성하는 함수"""
    api_key_middle = "zaSyBjhTX0EWpHXdvpYm9Dhk-fZFWLyU_"  # API 키 중간 부분
    user_input = st.text_input("비밀번호 6글자를 입력하세요(대소문자 구분!):", type="password")
    if len(user_input) != 6:
        st.error("비밀번호를 정확하게 입력해야 합니다.")
        return None
    api_key_prefix = user_input[:2]  # 입력값에서 앞 2글자 추출
    api_key_suffix = user_input[2:]  # 입력값에서 뒤 4글자 추출
    return api_key_prefix + api_key_middle + api_key_suffix

def main():
    """Streamlit 웹 애플리케이션의 메인 함수"""
    st.title("문서 비교 도구")

    # API 키 입력
    api_key = get_api_key()  

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
            "temperature"=0.1

            # 파일 또는 텍스트 입력 선택
            if prior_file and later_file:
                prior_text = read_file(prior_file)  # 파일 객체 전달
                later_text = read_file(later_file)  # 파일 객체 전달
            elif prior_text_input and later_text_input:
                prior_text = prior_text_input
                later_text = later_text_input
            else:
                st.error("두 텍스트를 모두 입력하거나 파일을 업로드하세요.")
                return

            if prior_text is None or later_text is None:
                return

            with st.spinner("텍스트 전처리 중..."):
                processed_prior_text = process_text_with_gemini(prior_text, model, additional_instructions)
                processed_later_text = process_text_with_gemini(later_text, model, additional_instructions)

            if processed_prior_text is None or processed_later_text is None:
                return

            with st.spinner("비교 중..."):
                comparison_result = compare_texts(processed_prior_text, processed_later_text, model)

            if comparison_result is None:
                return

            st.subheader("비교 결과:")
            st.text(comparison_result)

        except Exception as e:
            st.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
