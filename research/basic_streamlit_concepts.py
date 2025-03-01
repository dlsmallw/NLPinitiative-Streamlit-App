import streamlit as st
import pandas as pd
import numpy as np

st.write("Basic Dataframe Display")
dataframe = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})
st.write(dataframe)

st.write("Numpy (Randomized) Dataframe Display")
dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)

st.write("Use of Styling to highlight specific cells")
dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))

st.write('Displaying Charts')
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

st.write('Displaying Maps')
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)