import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter

# Title and Info
st.title("🛒 Grocery Recommendation App")
st.info("An app to explore grocery recommendation using association rule mining.")

# Load and Display Data
with st.expander("Dataset"):
    df = pd.read_csv('https://raw.githubusercontent.com/eymenslimani/data/refs/heads/main/Groceries_dataset.csv')
    st.write("**Raw Data**")
    st.write(df)

    # Preprocess Data
    transaction_data = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(lambda x: list(x)).tolist()
    te = TransactionEncoder()
    te_array = te.fit(transaction_data).transform(transaction_data)
    transaction_df = pd.DataFrame(te_array, columns=te.columns_)
    st.write("**Processed Transactions**")
    st.write(transaction_df)

# Association Rule Mining
min_support = st.sidebar.slider("Minimum Support", 0.0001, 0.05, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.7)

frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Display Rules
with st.expander("Association Rules"):
    st.write("**Frequent Itemsets**")
    st.write(frequent_itemsets)
    st.write("**Association Rules**")
    st.write(rules)

# Input for Predictions
with st.sidebar:
    st.header("Input for Recommendations")
    antecedent = st.multiselect("Select items", transaction_df.columns.tolist())
    top_n = st.slider("Top N Recommendations", 1, 10, 3)

def make_prediction(antecedent, rules, top_n):
    matching_rules = rules[rules['antecedents'].apply(lambda x: set(antecedent).issubset(x))]
    top_rules = matching_rules.sort_values(by='confidence', ascending=False).head(top_n)
    predictions = top_rules['consequents'].tolist()
    return [', '.join(list(pred)) for pred in predictions]

if antecedent:
    recommendations = make_prediction(antecedent, rules, top_n)
    st.sidebar.write("**Recommendations**")
    st.sidebar.write(recommendations)

# Visualization
with st.expander("Visualizations"):
    st.write("**Graph of Rules**")
    G = nx.DiGraph()
    for _, rule in rules.head(10).iterrows():
        for antecedent in rule['antecedents']:
            for consequent in rule['consequents']:
                G.add_edge(antecedent, consequent, weight=rule['lift'])
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=10, node_size=2000, 
            edge_color='gray', width=2)
    st.pyplot(plt)

    st.write("**Heatmap of Rules (Support vs Lift)**")
    if not rules.empty:
        plt.figure(figsize=(10, 6))
        heatmap_data = rules[['support', 'confidence', 'lift']].head(10)
        st.write(heatmap_data)  # Display DataFrame for clarity
        plt.barh(range(len(heatmap_data)), heatmap_data['lift'], color='skyblue')
        plt.xlabel('Lift')
        plt.title('Lift for Top Rules')
        st.pyplot(plt)
