import streamlit as st
import nest_asyncio
import pandas as pd
import os

from htbuilder import span, div
from loguru import logger
from annotated_text import annotation
from scripts.predict import InferenceHandler
from huggingface_hub import snapshot_download

from scripts.config import DATASET_REPO

nest_asyncio.apply()
st.set_page_config(layout='wide')
rc = None

def load_history(parent_elem):
    """Loads the history of results from inference for previous inputs made by the user.

    Parameters
    ----------
    parent_elem : DeltaGenerator
        The Streamlit UI element that contains the history data.
    """

    with parent_elem:
        if len(st.session_state.results) == 0:
            st.markdown(
                f"<p style='text-align: center; font-weight: bold; font-style: italic; font-size: 1.5vw;'>No History</p>", 
                unsafe_allow_html=True
            )
        else:
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

@st.cache_data
def load_inference_handler(api_token: str) -> InferenceHandler | None:
    """Loads an instance of the InferenceHandler class once a token is entered.

    Parameters
    ----------
    api_token: str
        The Hugging Face read/write token used for retrieving the binary classification and multilabel regression model tensor files.

    Returns
    -------
    InferenceHandler | None
        Returns an instance of the InferenceHandler class if a valid token is entered, otherwise returns None.
    """

    try:
        return InferenceHandler(api_token)
    except:
        return None

def build_result_tree(parent_elem, results: dict):
    """Loads the history of results from inference for previous inputs made by the user.

    Parameters
    ----------
    parent_elem : DeltaGenerator
        The Streamlit UI element to post the data to.
    results : dict
        The resulting data from performing inference.
    """

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
            'classification': f':red[{bin_class}]' if pred_class else f':green[{bin_class}]',
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
        result_hdr = ':red[Detected Discriminatory Sentiment]' if discriminatory_sentiment else ':green[No Discriminatory Sentiment Detected]'
        st.markdown(f'### Results - {result_hdr}')
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

                            classification = sent['classification']
                            st.markdown(f'##### Classification - {classification}')

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
                    sent = sent_details[0]
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
def analyze_text(input: str):
    """Performs infernce on the entered text using the InferenceHandler.
    
    Parameters
    ----------
    input : str
        The text to analyze.
    """
    if ih:
        res = None
        with rc:
            with st.spinner("Processing...", show_time=True) as spnr:
                # time.sleep(5)
                res = ih.classify_text(input)
                del spnr

        if res is not None:
            st.session_state.results.append(res)
            build_result_tree(rc, res)

@st.cache_data
def load_datasets(_parent_elem, api_token: str):
    if api_token is None or len(api_token) == 0:
        raise Exception()

    cache_path = snapshot_download(repo_id=DATASET_REPO, repo_type='dataset', token=api_token)
    ds_record = pd.read_csv(os.path.join(cache_path, 'dataset_record.csv'))
    
    raw_ds_path = os.path.join(cache_path, 'raw')
    interim_ds_path = os.path.join(cache_path, 'interim')
    processed_ds_path = os.path.join(cache_path, 'processed')

    with _parent_elem:
        st.markdown(f'### Disclaimer')
        st.markdown("> The datasets displayed contain content that may be highly discriminatory or offensive in nature. Viewer discretion is advised. This content is presented solely for analysis, research, or educational purposes and does not reflect the views or values of the creators or maintainers of this application.")
        st.markdown('<hr style="margin: 0 0 0.5em 0;">', unsafe_allow_html=True)

        if os.path.exists(os.path.join(processed_ds_path, 'NLPinitiative_Master_Dataset.csv')):
            master_df = pd.read_csv(os.path.join(processed_ds_path, 'NLPinitiative_Master_Dataset.csv'))

            if len(master_df) > 0:
                st.markdown(f'### NLPinitiative Master Dataset')
                with st.expander(label='Master Dataset'):
                    st.dataframe(master_df)

        if len(ds_record) > 0:
            for _, row in ds_record.iterrows():
                try:
                    ds_id = row['Dataset ID']
                    ds_ref_url = row['Dataset Reference URL']
                    raw_fn = row['Raw Dataset Filename']
                    norm_fn = row['Converted Filename']

                    raw_df = pd.read_csv(os.path.join(raw_ds_path, raw_fn))
                    norm_df = pd.read_csv(os.path.join(interim_ds_path, norm_fn))

                    st.markdown('<hr style="margin: 0 0 0.5em 0;">', unsafe_allow_html=True)
                    st.markdown(f'#### {ds_id} - [Link to Dataset Source]({ds_ref_url})')
                    with st.expander(label='Dataset'):
                        st.markdown(f'###### Raw Dataset')
                        st.dataframe(raw_df)
                        st.markdown(f'###### Normalized Dataset')
                        st.dataframe(norm_df)
                    
                except Exception as e:
                    logger.error(f'{e}')
        else:
            st.markdown(
                f"<p style='text-align: center; font-weight: bold; font-style: italic; font-size: 1.5vw;'>No Datasets to Display</p>", 
                unsafe_allow_html=True
            )

#===========================================================================================================================================

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
tab4 = st.empty()
tab3 = st.empty()

tab1, tab2, tab3, tab4 = st.tabs(['Classifier', 'Input History', 'Datasets', 'About This App'])

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
    hist_container = st.container(border=True)
    try:
        load_history(hist_container)
    except:
        hist_container.markdown(
            f"<p style='text-align: center; font-weight: bold; font-style: italic; font-size: 1.5vw;'>No History</p>", 
            unsafe_allow_html=True
        )

with tab3:
    ds_container = st.container(border=True)
    try:
        load_datasets(ds_container, API_KEY)
    except Exception as e:
        logger.error(f'{e}')
        ds_container.markdown(
            f"<p style='text-align: center; font-weight: bold; font-style: italic; font-size: 1.5vw;'>No Datasets to Display</p>", 
            unsafe_allow_html=True
        )

with tab4:
    st.markdown(
        f"""
        ## About
        The NLPinitiative Discriminatory Text Classifier is an advanced 
        natural language processing tool designed to detect and flag potentially 
        discriminatory or harmful language. By analyzing text for biased, offensive, 
        or exclusionary content, this classifier helps promote more inclusive and 
        respectful communication. Simply enter your text below, and the model will 
        assess it based on linguistic patterns and context. While the tool provides 
        valuable insights, we encourage users to review flagged content thoughtfully 
        and consider context when interpreting results.

        The application utilizes two NLP models: a fine-tuned binary classifier for classifying input as 
        Discriminatory or Non-Discriminatory and a fine-tuned multilabel regression model for assessing 
        the likelihood of specific categories of discrimination (Gender, Race, Sexuality, Disability, Religion 
        and Unspecified). The base model used for both fine-tuned models is the pretrained 
        [BERT](https://doi.org/10.48550/arXiv.1810.04805) (Bidirectional Encoder Representations from Transformers) 
        model.
        """
    )