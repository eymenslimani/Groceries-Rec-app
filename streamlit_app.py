import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter

# Title and App Info
st.title("🛒 Grocery Recommendation App")
st.info("This app provides product recommendations using association rule mining techniques.")

# Load Dataset
data_url = 'https://raw.githubusercontent.com/eymenslimani/data/refs/heads/main/Groceries_dataset.csv'
data = pd.read_csv(data_url)
data.dropna(inplace=True)

# Display Dataset
with st.expander("Dataset Overview"):
    st.write("**Sample of Raw Data**")
    st.write(data.head())

# Preprocess Data
trans_data = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(lambda x: ','.join(x)).reset_index()
trans_data = trans_data.drop(columns=['Member_number', 'Date'])
trans_data.columns = ['itemDescription']
transactions = trans_data['itemDescription'].str.split(',')

# Encode Data for Model
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(te_array, columns=te.columns_)

# Generate Frequent Itemsets and Rules
def generate_rules(transaction_df, min_support, min_confidence):
    frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    return frequent_itemsets, rules

# User Input Section
st.sidebar.header("Input Parameters")
min_support = st.sidebar.slider("Minimum Support", 0.0001, 0.05, 0.001, step=0.0001)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.7, step=0.1)
user_input = st.sidebar.text_input("Enter product(s) (comma-separated):", "bottled water")

# Generate Rules
frequent_itemsets, rules = generate_rules(transaction_df, min_support, min_confidence)

# Display Frequent Itemsets
with st.expander("Frequent Itemsets"):
    st.write(frequent_itemsets)

# Display Rules
with st.expander("Association Rules"):
    st.write(rules)

# Recommendation Based on User Input
def make_prediction(antecedents, rules, top_n=5):
    antecedent_set = set(antecedents)
    matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(antecedent_set))]
    top_rules = matching_rules.sort_values(by='confidence', ascending=False).head(top_n)
    predictions = top_rules['consequents'].tolist()
    formatted_predictions = [', '.join(list(consequent)) for consequent in predictions]
    return formatted_predictions

if user_input:
    user_products = [item.strip() for item in user_input.split(',')]
    predictions = make_prediction(user_products, rules)
    st.subheader("Recommendations")
    if predictions:
        st.write(f"For the product(s): {', '.join(user_products)}, we recommend:")
        for i, pred in enumerate(predictions, 1):
            st.write(f"{i}. {pred}")
    else:
        st.write("No recommendations found for the given input.")

# Visualizations
with st.expander("Visualizations"):
    st.subheader("Item Frequency Plot")
    def plot_item_frequency(transactions, top_n=20):
        item_counts = Counter(item for sublist in transactions for item in sublist)
        most_common_items = item_counts.most_common(top_n)
        items, counts = zip(*most_common_items)
        plt.figure(figsize=(10, 6))
        plt.bar(items, counts, color=plt.cm.Pastel2(range(len(items))))
        plt.xticks(rotation=90)
        plt.title("Top Items by Frequency")
        plt.ylabel("Frequency")
        st.pyplot(plt)

    plot_item_frequency(transactions)

    st.subheader("Top Association Rules Graph")
    G = nx.DiGraph()
    top_rules = rules.nlargest(10, 'confidence')
    for _, rule in top_rules.iterrows():
        for antecedent in rule['antecedents']:
            for consequent in rule['consequents']:
                G.add_edge(antecedent, consequent, weight=rule['lift'])

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=10, node_size=2000, 
            edge_color=weights, edge_cmap=plt.cm.viridis, width=2)
    plt.title("Graph-Based Visualization of Top 10 Rules")
    st.pyplot(plt)
