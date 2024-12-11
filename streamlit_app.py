import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# App title and introduction
st.title("🍭 Grocery Recommendation App")
st.markdown("An interactive app to explore grocery recommendations using association rule mining.")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/eymenslimani/data/refs/heads/main/Groceries_dataset.csv')
    data.dropna(inplace=True)
    return data

data = load_data()

# Dataset preview
with st.expander("View Dataset"):
    st.dataframe(data.head())

# Preprocess transactions
def preprocess_transactions(data):
    trans_data = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(lambda x: ','.join(x)).reset_index()
    trans_data = trans_data.drop(columns=['Member_number', 'Date'])
    transactions = trans_data['itemDescription'].str.split(',')
    return transactions

transactions = preprocess_transactions(data)

# Encode transactions
@st.cache_data
def encode_transactions(transactions):
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_array, columns=te.columns_)

transaction_df = encode_transactions(transactions)

# Generate frequent itemsets and rules
def generate_rules(transaction_df, min_support, min_confidence):
    frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    return rules

min_support = st.sidebar.slider("Minimum Support", 0.0001, 0.05, 0.001, 0.0001)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.7, 0.1)
rules = generate_rules(transaction_df, min_support, min_confidence)

# Input product(s) and generate recommendations
st.subheader("Product Recommendation")
all_products = sorted(transaction_df.columns)
user_input = st.multiselect("Select a product or multiple products:", all_products)

if user_input:
    antecedent_set = set(user_input)
    matching_rules = rules[rules['antecedents'].apply(lambda x: antecedent_set.issubset(x))]
    recommendations = matching_rules.sort_values(by='confidence', ascending=False).head(5)['consequents']

    if not recommendations.empty:
        st.write("**Recommended Products:**")
        for rec in recommendations:
            st.write(", ".join(rec))
    else:
        st.warning("No recommendations found for the selected product(s).")

# Visualize association rules
if not rules.empty:
    st.subheader("Top Rules Visualization")
    top_rules = rules.nlargest(10, 'confidence')

    # Network graph visualization
    G = nx.DiGraph()
    for _, rule in top_rules.iterrows():
        for antecedent in rule['antecedents']:
            for consequent in rule['consequents']:
                G.add_edge(antecedent, consequent, weight=rule['lift'])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=10, node_size=2000, 
            edge_color='gray', width=2)
    st.pyplot(plt)

    # Heatmap visualization
    st.write("**Heatmap of Support, Confidence, and Lift:**")
    heatmap_data = top_rules[['support', 'confidence', 'lift']]
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)
