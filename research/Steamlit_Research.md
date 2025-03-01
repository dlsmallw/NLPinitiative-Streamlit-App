# Streamlit Research:
### Creating the application:
 - Import Streamlit library

```
import streamlit as st
```

 - Running the Streamlit app:
    - This will open a separate browser window running the application locally
```
streamlit run <app_name>.py
```


### State driven UI:
 - Elements can be loaded into variables and those variables can be used to modify the specified element:
    ```
    text_element_state = st.text('Element State Test')  ## Initializes the element with this text
    text_element_state.text('New Element State Test')   ## Updates the displayed text
    ```

 - Can also using caching:
    - Minimizes overhead of running functions everytime the application reloads by checking the passed values to see if they have changed and only running the method if they have
        ```
        @st.cache_data
        def load_data(args):
            ## Some logic

        text_element_state.text('Some data: (using st.cache_data)') ## This will appear immediately upon saving
        ```
 - There are some limitations with using the cache mechanism:
    - Will not work with functions with internal numerical randomization (with regards to calculations)
    - Scope of the validation is within the current working directory
    - Cached values are stored by reference, so it is undesirable to mutate the values
 - st.write() function:
    - Can pass more complex data types that will be displayed in an interactable format
        - i.e., dataframes can be passed and displayed automatically as a table
 - Additional functionality that is available:
    - Draw histograms by using numpy:
        ```
        st.subheader('Number of pickups by hour')
        hist_values = np.histogram(
            data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
        st.bar_chart(hist_values)
        ```
    - Plot data on map:
        ```
        hour_to_filter = 17
        filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
        st.subheader(f'Map of all pickups at {hour_to_filter}:00')
        st.map(filtered_data)
        ```
    - Filter data using slider:
        ```
        hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
        ```
    - Using buttons for toggling data:
        ```
        if st.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(data)
        ```