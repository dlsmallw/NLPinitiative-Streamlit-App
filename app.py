import streamlit as st
import pandas as pd
from annotated_text import annotated_text, annotation
import time
from random import randint, uniform
from scripts.predict import InferenceHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
st.write(ROOT)
MODELS_DIR = ROOT / 'models'
BIN_MODEL_PATH = MODELS_DIR / 'binary_classification'
ML_MODEL_PATH = MODELS_DIR / 'multilabel_regression'

history_df = pd.DataFrame(data=[], columns=['Text', 'Classification', 'Gender', 'Race', 'Sexuality', 'Disability', 'Religion', 'Unspecified'])
ih = InferenceHandler(BIN_MODEL_PATH, ML_MODEL_PATH)

def extract_data(json_obj):
    row_data = []

    row_data.append(json_obj['raw_text'])
    row_data.append(json_obj['text_sentiment'])
    cat_dict = json_obj['category_sentiments']
    for cat in cat_dict.keys():
        raw_val = cat_dict[cat]
        val = f'{raw_val * 100: .2f}%' if raw_val is not None else 'N/A'
        row_data.append(val)
    
    return row_data

def load_history():
    for result in st.session_state.results:
        history_df.loc[len(history_df)] = extract_data(result)

def output_results(res):
    label_dict = {
        'Gender': '#4A90E2',
        'Race': '#E67E22',
        'Sexuality': '#3B9C5A',  
        'Disability': '#8B5E3C',
        'Religion': '#A347BA',  
        'Unspecified': '#A0A0A0'
    }

    with rc:
        st.markdown('### Results')
        with st.container(border=True):
            at_list = []
            if res['numerical_sentiment'] == 1:
                for entry in res['category_sentiments'].keys():
                    val = res['category_sentiments'][entry]
                    if val > 0.0:
                        perc = val * 100
                        at_list.append((entry, f'{perc:.2f}%', label_dict[entry]))

            st.markdown(f"#### Text - *\"{res['raw_text']}\"*")
            st.markdown(f"#### Classification - {':red' if res['numerical_sentiment'] == 1 else ':green'}[{res['text_sentiment']}]")

            if len(at_list) > 0:
                annotated_text(at_list)
                        

# def test_results(text):
#     test_val = int(randint(0, 1))
#     res_obj = {
#             'raw_text': text,
#             'text_sentiment': 'Discriminatory' if test_val == 1 else 'Non-Discriminatory',
#             'numerical_sentiment': test_val,
#             'category_sentiments': {
#                 'Gender': None if test_val == 0 else uniform(0.0, 1.0),
#                 'Race': None if test_val == 0 else uniform(0.0, 1.0),
#                 'Sexuality': None if test_val == 0 else uniform(0.0, 1.0),  
#                 'Disability': None if test_val == 0 else uniform(0.0, 1.0),
#                 'Religion': None if test_val == 0 else uniform(0.0, 1.0),  
#                 'Unspecified': None if test_val == 0 else uniform(0.0, 1.0)
#             }
#         }
#     return res_obj


def analyze_text(text):
    res = None
    with rc:
        with st.spinner("Processing...", show_time=True) as spnr:
            time.sleep(5)
            res = ih.classify_text(text)
            del spnr

    if res is not None:
        st.session_state.results.append(res)
        history_df.loc[-1] = extract_data(res)
        output_results(res)

st.title('NLPinitiative Text Classifier')
tab1, tab2 = st.tabs(['Classifier', 'About This App'])

if "results" not in st.session_state:
    st.session_state.results = []
    
load_history()

with tab1:
    "Text Classifier for determining if entered text is discriminatory (and the categories of discrimination) or Non-Discriminatory."

    with st.container():
        with st.expander('History'):
            st.write(history_df)

        rc = st.container()

    text_form = st.form(key='classifier', clear_on_submit=True, enter_to_submit=True)
    with text_form:
        text_area = st.text_area('Enter text to classify')
        form_btn = st.form_submit_button('submit')

        if entry := text_area:
            analyze_text(entry)


with tab2:
    st.markdown(
    """The NLPinitiative Discriminatory Text Classifier is an advanced 
    natural language processing tool designed to detect and flag potentially 
    discriminatory or harmful language. By analyzing text for biased, offensive, 
    or exclusionary content, this classifier helps promote more inclusive and 
    respectful communication. Simply enter your text below, and the model will 
    assess it based on linguistic patterns and context. While the tool provides 
    valuable insights, we encourage users to review flagged content thoughtfully 
    and consider context when interpreting results."""
)