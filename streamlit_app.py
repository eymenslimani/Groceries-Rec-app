import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter

# Load and Display Data
st.title("🛒 Grocery Recommendation App")
st.info("An app to explore grocery recommendation using association rule mining.")

