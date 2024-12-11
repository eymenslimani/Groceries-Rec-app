import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter

# Title and App Info
st.title("🛒 Grocery Recommendation App")
st.info("An app to explore grocery recommendation using association rule mining.")

# Load Dataset
def load_data():
    url = 'https://raw.githubusercontent.com/eymenslimani/data/refs/heads/main/Groceries_dataset.csv'
    data = pd.read_csv(url)
    data.dropna(inplace=True)
    return data

data = load_data()

# Dataset Exploration
with st.expander("Dataset"):
    st.write("**Raw Data**")
    st.write(data.head())
    st.write("**Data Info**")
    st.write(data.info())

# Process Data
def preprocess_data(data):
    transaction_data = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(lambda x: ','.join(x)).reset_index()
    transaction_data = transaction_data.drop(columns=['Member_number', 'Date'])
    transaction_data.columns = ['itemDescription']
    transactions = transaction_data['itemDescription'].str.split(',')
    return transactions

transactions = preprocess_data(data)

# Encoding Transactions
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(te_array, columns=te.columns_)

# Visualization: Item Frequency
def plot_item_frequency(transactions, top_n=20):
    item_counts = Counter(item for sublist in transactions for item in sublist)
    most_common_items = item_counts.most_common(top_n)
    items, counts = zip(*most_common_items)
    plt.bar(items, counts, color=plt.cm.Pastel2(range(len(items))))
    plt.title("Absolute Item Frequency Plot")
    plt.xticks(rotation=90)
    plt.ylabel("Frequency")
    st.pyplot(plt)

with st.expander("Item Frequency"):
    st.subheader("Top 20 Items")
    plot_item_frequency(transactions)

# Generate Frequent Itemsets and Rules
def generate_rules(df, method='apriori', min_support=0.01, min_confidence=0.2):
    if method == 'apriori':
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    return rules

method = st.sidebar.selectbox("Select Algorithm", ["Apriori", "FP-Growth"])
min_support = st.sidebar.slider("Minimum Support", 0.001, 0.05, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.2)

rules = generate_rules(transaction_df, method.lower(), min_support, min_confidence)

with st.expander("Generated Rules"):
    st.write(rules.head())

# Prediction Function
def make_prediction(antecedent, rules, top_n=3):
    matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(antecedent))]
    top_rules = matching_rules.sort_values(by='confidence', ascending=False).head(top_n)
    predictions = top_rules['consequents'].tolist()
    formatted_predictions = [', '.join(list(consequent)) for consequent in predictions]
    return formatted_predictions

antecedent_input = st.text_input("Enter items (comma-separated):", "bottled water")
if antecedent_input:
    antecedent = set(antecedent_input.split(','))
    predictions = make_prediction(antecedent, rules)
    st.write(f"**Predicted Consequents for {antecedent}:** {predictions}")

# Visualization: Graph of Rules
def plot_graph(rules, top_n=10):
    G = nx.DiGraph()
    top_rules = rules.nlargest(top_n, 'confidence')
    for _, rule in top_rules.iterrows():
        for antecedent in rule['antecedents']:
            for consequent in rule['consequents']:
                G.add_edge(antecedent, consequent, weight=rule['lift'])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1, seed=42)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=10, node_size=2000, 
            edge_color=weights, edge_cmap=plt.cm.viridis, width=2)
    st.pyplot(plt)

with st.expander("Rule Graph"):
    st.subheader("Graph Visualization of Rules")
    plot_graph(rules)
