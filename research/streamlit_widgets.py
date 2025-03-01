import streamlit as st
import pandas as pd
import numpy as np

st.write('Sliders:')
slider_val = st.slider('Data Slider') 
st.write(slider_val, 'squared is', slider_val * slider_val)

st.write('Reference by Widget Name:')
st.text_input("Text Entry", key="text")
st.session_state.text

st.write('Checkboxes:')
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

st.write('Selectboxes:')
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Favorite Number?',
     df['first column'])

'Selected Number: ', option