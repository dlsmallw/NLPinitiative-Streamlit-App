import streamlit as st
import pandas as pd
import numpy as np
import time


## Sidebar Usage:
ex_select_box = st.sidebar.selectbox(
    'Favorite Music Genre:',
    ('Rock', 'Reggae', 'Bluegrass')
)

ex_slider = st.sidebar.slider(
    'Age Range:',
    0, 100, (25, 75)
)

## Column Usage:
lc, rc = st.columns(2)
## Columns like sidebars:
lc.button('Example - Left Column Button')

## Using with blocks:
with rc:
    choice = st.radio(
        'Favorite Color:',
        ("Red", "Blue", "Green", "Magenta"))
    st.write(f"{choice} is your favorite color!" if choice != 'Magenta' else f'{choice} is not a natural color!')

## Progress Bars:
'Loading...'

percentage = st.empty()
bar = st.progress(0)

for i in range(100):
    # Update the progress bar with each iteration.
    percentage.text(f'{i+1}%')
    bar.progress(i + 1)
    time.sleep(0.1)
'Loading complete!'

## Sessions:
if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again") 