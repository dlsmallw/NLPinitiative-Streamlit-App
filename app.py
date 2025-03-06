import streamlit as st
import pandas as pd
from annotated_text import annotated_text
import time
from scripts.predict import InferenceHandler

history_df = pd.DataFrame(data=[], columns=['Text', 'Classification', 'Gender', 'Race', 'Sexuality', 'Disability', 'Religion', 'Unspecified'])
rc = None
ih = None
entry = None

@st.cache_data
def load_inference_handler(api_token):
    ih = InferenceHandler(api_token)

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

@st.cache_data
def analyze_text(text):
    if ih:
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

st.sidebar.write("")
API_KEY = st.sidebar.text_input(
    "Enter your HuggingFace API Token",
    help="You can get your free API token in your settings page: https://huggingface.co/settings/tokens",
    type="password",
)

try:
    if API_KEY is not None and len(API_KEY) > 0:
        ih = InferenceHandler(API_KEY)
except:
    ih = None
    st.error('Invalid Token')

tab1, tab2 = st.tabs(['Classifier', 'About This App'])

if "results" not in st.session_state:
    st.session_state.results = []
    
load_history()

with tab1:
    "Text Classifier for determining if entered text is discriminatory (and the categories of discrimination) or Non-Discriminatory."

    hist_container = st.container()
    hist_expander = hist_container.expander('History')
    rc = st.container()
    
    text_form = st.form(key='classifier', clear_on_submit=True, enter_to_submit=True)
    with text_form:
        text_area = st.text_area('Enter text to classify', value='', disabled=True if ih is None else False)
        form_btn = st.form_submit_button('submit', disabled=True if ih is None else False)

        if entry := text_area:
            st.write(f'TEXT AREA: {entry}')
            if entry and len(entry) > 0:
                analyze_text(entry)
                entry = None

    with hist_expander:
        st.dataframe(history_df)

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