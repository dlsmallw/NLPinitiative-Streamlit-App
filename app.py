import streamlit as st
import nest_asyncio
import pandas as pd
from annotated_text import annotation
from scripts.predict import InferenceHandler
from htbuilder import span, div

nest_asyncio.apply()

rc = None

@st.cache_data
def load_inference_handler(api_token):
    try:
        return InferenceHandler(api_token)
    except:
        return None

@st.cache_data
def extract_data(json_obj):
    row_data = []

    row_data.append(json_obj['text_input'])
    row_data.append(json_obj['text_sentiment'])
    cat_dict = json_obj['category_sentiments']
    for cat in cat_dict.keys():
        raw_val = cat_dict[cat]
        val = f'{raw_val * 100: .2f}%' if raw_val is not None else 'N/A'
        row_data.append(val)
    
    return row_data

def load_history(parent_elem):
    with parent_elem:
        for idx, result in enumerate(st.session_state.results):
            text = result['text_input']
            discriminatory = False

            data = []
            for sent_item in result['results']:
                sentence = sent_item['sentence']
                bin_class = sent_item['binary_classification']['classification']
                pred_class = sent_item['binary_classification']['prediction_class']
                ml_regr = sent_item['multilabel_regression']

                row_data = [sentence, bin_class]
                if pred_class == 1:
                    discriminatory = True
                    for cat in ml_regr.keys():
                        perc = ml_regr[cat] * 100
                        row_data.append(f'{perc:.2f}%')
                else:
                    for i in range(6):
                        row_data.append(None)

                data.append(row_data)
            df = pd.DataFrame(data=data, columns=['Sentence', 'Binary Classification', 'Gender', 'Race', 'Sexuality', 'Disability', 'Religion', 'Unspecified'])

            with st.expander(label=f'Entry #{idx+1}', icon='🔴' if discriminatory else '🟢'):
                st.markdown('<hr style="margin: 0.5em 0 0 0;">', unsafe_allow_html=True)
                st.markdown(
                    f"<p style='text-align: center; font-weight: bold; font-style: italic; font-size: medium;'>\"{text}\"</p>", 
                    unsafe_allow_html=True
                )
                st.markdown('<hr style="margin: 0 0 0.5em 0;">', unsafe_allow_html=True)
                st.markdown('##### Sentence Breakdown:')
                st.dataframe(df)


def build_result_tree(parent_elem, results):
    label_dict = {
        'Gender': '#4A90E2',
        'Race': '#E67E22',
        'Sexuality': '#3B9C5A',  
        'Disability': '#8B5E3C',
        'Religion': '#A347BA',  
        'Unspecified': '#A0A0A0'
    }

    discriminatory_sentiment = False

    sent_details = []
    for result in results['results']:
        sentence = result['sentence']
        bin_class = result['binary_classification']['classification']
        pred_class = result['binary_classification']['prediction_class']
        ml_regr = result['multilabel_regression']

        sent_res = {
            'sentence': sentence,
            'classification': f'{":red" if pred_class else ":green"}[{bin_class}]',
            'annotated_categories': []
        }

        if pred_class == 1:
            discriminatory_sentiment = True
            at_list = []
            for entry in ml_regr.keys():
                val = ml_regr[entry]
                if val > 0.0:
                    perc = val * 100
                    at_list.append(annotation(body=entry, label=f'{perc:.2f}%', background=label_dict[entry]))
            sent_res['annotated_categories'] = at_list
        sent_details.append(sent_res)

    with parent_elem:
        st.markdown(f'### Results - {':red[Detected Discriminatory Sentiment]' if discriminatory_sentiment else ':green[No Discriminatory Sentiment Detected]'}')
        with st.container(border=True):
            st.markdown('<hr style="margin: 0.5em 0 0 0;">', unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align: center; font-weight: bold; font-style: italic; font-size: large;'>\"{results['text_input']}\"</p>", 
                unsafe_allow_html=True
            )
            st.markdown('<hr style="margin: 0 0 0.5em 0;">', unsafe_allow_html=True)

            if discriminatory_sentiment:
                if (len(results['results']) > 1):
                    st.markdown('##### Sentence Breakdown:')
                    for idx, sent in enumerate(sent_details):
                        with st.expander(label=f'Sentence #{idx+1}', icon='🔴' if len(sent['annotated_categories']) > 0 else '🟢', expanded=True):
                            st.markdown('<hr style="margin: 0.5em 0 0 0;">', unsafe_allow_html=True)
                            st.markdown(
                                f"<p style='text-align: center; font-weight: bold; font-style: italic; font-size: large;'>\"{sent['sentence']}\"</p>", 
                                unsafe_allow_html=True
                            )
                            st.markdown('<hr style="margin: 0 0 0.5em 0;">', unsafe_allow_html=True)
                            st.markdown(f'##### Classification - {sent['classification']}')

                            if len(sent['annotated_categories']) > 0:
                                st.markdown(
                                    div(    
                                        span(' ' if idx != 0 else '')[
                                            item
                                        ] for idx, item in enumerate(sent['annotated_categories'])
                                    ),
                                    unsafe_allow_html=True
                                )
                                st.markdown('\n')
                else:
                    st.markdown(f"#### Classification - {sent['classification']}")
                    if len(sent['annotated_categories']) > 0:
                        st.markdown(
                            div(    
                                span(' ' if idx != 0 else '')[
                                    item
                                ] for idx, item in enumerate(sent['annotated_categories'])
                            ),
                            unsafe_allow_html=True
                        )
                        st.markdown('\n')

@st.cache_data
def analyze_text(text):
    if ih:
        res = None
        with rc:
            with st.spinner("Processing...", show_time=True) as spnr:
                # time.sleep(5)
                res = ih.classify_text(text)
                del spnr

        if res is not None:
            st.session_state.results.append(res)
            build_result_tree(rc, res)

st.title('NLPinitiative Text Classifier')

st.sidebar.write("")
API_KEY = st.sidebar.text_input(
    "Enter your HuggingFace API Token",
    help="You can get your free API token in your settings page: https://huggingface.co/settings/tokens",
    type="password",
)
ih = load_inference_handler(API_KEY)

tab1 = st.empty()
tab2 = st.empty()
tab3 = st.empty()

tab1, tab2, tab3 = st.tabs(['Classifier', 'Input History', 'About This App'])

if "results" not in st.session_state:
    st.session_state.results = []

with tab1:
    "Text Classifier for determining if entered text is discriminatory (and the categories of discrimination) or Non-Discriminatory."

    rc = st.container()
    text_form = st.form(key='classifier', clear_on_submit=True, enter_to_submit=True)
    with text_form:
        entry = None
        text_area = st.text_area('Enter text to classify', value='', disabled=True if ih is None else False)
        form_btn = st.form_submit_button('submit', disabled=True if ih is None else False)
        if form_btn and text_area is not None and len(text_area) > 0:
            analyze_text(text_area)

with tab2:
    load_history(tab2)

with tab3:
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