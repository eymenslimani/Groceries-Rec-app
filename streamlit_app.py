# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

st.set_page_config(page_title="Groceries Recommendation System", layout="wide")
st.title("Groceries Recommendation System")      # Sidebar for inputs
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Transaction Data (CSV)", type=["csv"])
min_support = st.sidebar.slider("Minimum Support", 0.0001, 0.05, 0.001, step=0.0001)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.7, step=0.1)
algorithm = st.sidebar.selectbox("Select Algorithm", ["Apriori", "FPGrowth"]) if uploaded_file:
    # Read CSV
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", data.head())

    # Preprocess data
    data.dropna(inplace=True)
    transactions = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(lambda x: list(x)).tolist()

    # One-hot encode transactions
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_array, columns=te.columns_)
    st.write("### Encoded Transactions", transaction_df.head())
else:
    st.warning("Please upload a transaction dataset.") 
    if uploaded_file:
    if algorithm == "Apriori":
        frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True, max_len=10)
    else:  # FPGrowth
        from mlxtend.frequent_patterns import fpgrowth
        frequent_itemsets = fpgrowth(transaction_df, min_support=min_support, use_colnames=True, max_len=10)

    st.write("### Frequent Itemsets", frequent_itemsets)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    st.write("### Association Rules", rules) 
    if uploaded_file and not rules.empty:
    st.header("Make Predictions")
    antecedent = st.text_input("Enter antecedent items (comma-separated):").split(",")

    if antecedent:
        antecedent_set = set(antecedent)
        matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(antecedent_set))]
        predictions = matching_rules.sort_values(by="confidence", ascending=False).head(3)
        st.write("### Predicted Consequents", predictions[['consequents', 'confidence']]) 
        def plot_item_frequency(transactions):
    item_counts = Counter(item for sublist in transactions for item in sublist)
    most_common_items = item_counts.most_common(20)

    items, counts = zip(*most_common_items)
    plt.figure(figsize=(10, 5))
    plt.bar(items, counts, color=plt.cm.Pastel2(range(len(items))))
    plt.title("Top 20 Frequent Items")
    plt.xticks(rotation=90)
    st.pyplot(plt)

plot_item_frequency(transactions)  if not rules.empty:
    top10_rules = rules.nlargest(10, "confidence")
    plt.figure(figsize=(10, 6))
    sns.heatmap(top10_rules[["support", "confidence", "lift"]], annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)  if not rules.empty:
    G = nx.DiGraph()
    for _, rule in top10_rules.iterrows():
        for antecedent in rule["antecedents"]:
            for consequent in rule["consequents"]:
                G.add_edge(antecedent, consequent, weight=rule["lift"])

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue", font_size=10)
    st.pyplot(plt)
