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

df = pd.read_csv('Groceries_dataset.csv')
st.write("**Raw Data**")
st.write(df)

# Data Preprocessing
transaxtionData = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(lambda x: ','.join(x)).reset_index()
transaxtionData = transaxtionData.drop(columns=['Member_number', 'Date'])
transaxtionData.columns = ['itemDescription']
transaxtionData.to_csv("assignment1s_itemslist.csv", index=False, quoting=0)

file_path ='assignment1s_itemslist.csv'
df = pd.read_csv(file_path, header=None)
df = df.drop(index=0).reset_index(drop=True)

transactions = df[0].str.split(',')

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
transaction_df = pd.DataFrame(te_array, columns=te.columns_)

# Generate Frequent Itemsets and Rules
frequent_itemsets = apriori(transaction_df, min_support=0.0001, use_colnames=True, low_memory=True,max_len=10)
rules = association_rules(frequent_itemsets, num_itemsets=len(transaction_df), metric='confidence', min_threshold=0.7)

# User Input
st.header("Enter Products:")
user_input = st.text_input("Enter product(s) separated by commas (e.g., 'whole milk', 'yogurt'):")
user_products = set(user_input.lower().strip().split(','))

# Prediction Function
def make_prediction(antecedent, rules, top_n=3):
    matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(antecedent))]
    top_rules = matching_rules.sort_values(by='confidence', ascending=False).head(top_n)
    predictions = top_rules['consequents'].tolist()
    formatted_predictions = [', '.join(list(consequent)) for consequent in predictions]
    return formatted_predictions

# Make Prediction
if user_input:
    predictions = make_prediction(user_products, rules)
    if predictions:
        st.success(f"Predicted Products: {predictions}")
    else:
        st.warning("No strong associations found for the given product(s).")
else:
    st.info("Please enter at least one product.")

# Visualization (Optional - Can be added as a separate section)
# ... (Add visualization code here, e.g., networkx graph, heatmap)

# Add some more informative sections
st.header("About the Model")
st.write("""
This app uses the Apriori algorithm to discover association rules within the grocery dataset. 
Association rules identify frequent itemsets and their relationships, allowing us to predict 
which products are likely to be purchased together.
""")

st.header("Disclaimer")
st.write("""
This is a simplified demonstration. Real-world applications would involve more complex 
data preprocessing, model tuning, and consideration of various factors like user history, 
promotions, and seasonality.
""")
