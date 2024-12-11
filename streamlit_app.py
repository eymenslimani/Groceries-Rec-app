import streamlit as st
import pandas as pd


st.title('Grocery Recommendation')
st.info('an app for grocery product recommendation')
df = pd.read_csv('https://raw.githubusercontent.com/eymenslimani/data/refs/heads/main/Groceries_dataset.csv')
df
