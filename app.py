import streamlit as st
from annotated_text import annotated_text, annotation
import time
from random import randint, uniform

results_container = None

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
        with st.chat_message(name = 'human', avatar=None):
            at_list = []
            if res['numerical_sentiment'] == 1:
                # st.markdown('##### Category Results:')
                for entry in res['category_sentiments'].keys():
                    if randint(0, 1) == 1:
                        val = res['category_sentiments'][entry]
                        perc = val * 100
                        at_list.append((entry, f'{perc:.2f}%', label_dict[entry]))

            st.markdown(f"#### Text - *\"{res['raw_text']}\"*")
            st.markdown(f"#### Classification - {':red' if res['numerical_sentiment'] == 1 else ':green'}[{res['text_sentiment']}]")

            if len(at_list) > 0:
                st.markdown('#### Categories: ')
                cols = st.columns([1, 15])
                with cols[1]:
                    for cat in at_list:
                        annotated_text(cat)

def test_results(text):
    test_val = int(randint(0, 1))
    res_obj = {
            'raw_text': text,
            'text_sentiment': 'Discriminatory' if test_val == 1 else 'Non-Discriminatory',
            'numerical_sentiment': test_val,
            'category_sentiments': {
                'Gender': None if test_val == 0 else uniform(0.0, 1.0),
                'Race': None if test_val == 0 else uniform(0.0, 1.0),
                'Sexuality': None if test_val == 0 else uniform(0.0, 1.0),  
                'Disability': None if test_val == 0 else uniform(0.0, 1.0),
                'Religion': None if test_val == 0 else uniform(0.0, 1.0),  
                'Unspecified': None if test_val == 0 else uniform(0.0, 1.0)
            }
        }
    return res_obj


def analyze_text(text):
    res = None
    with rc:
        with st.spinner("Processing...", show_time=True) as spnr:
            time.sleep(5)
            res = test_results(text)
            del spnr

    if res is not None:
        st.session_state.results.append(res)
        output_results(res)

pri_container = st.container()
cols = pri_container.columns([1, 8, 1])
cols[1].subheader('NLPinitiative - Discriminatory Text Classifier')

with pri_container.expander('About This Application'):
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

st.divider()

chat_container = st.container()
rc = chat_container.container(height=500)

if "results" not in st.session_state:
    st.session_state.results = []

with rc:
    for result in st.session_state.results:
        output_results(result)

if entry := chat_container.chat_input('Enter text to classify'):
    analyze_text(entry)