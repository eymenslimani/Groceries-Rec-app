import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# Page Configuration
st.set_page_config(page_title="Grocery Recommendation App", page_icon="🛒", layout="wide")

# Caching data loading for performance
@st.cache_data
def load_data():
    """
    Load and preprocess the grocery dataset
    """
    # Load the dataset from GitHub raw file
    data = pd.read_csv('https://raw.githubusercontent.com/eymenslimani/data/refs/heads/main/Groceries_dataset.csv')
    data.dropna(inplace=True)

    # Prepare transactions
    transactionData = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(lambda x: ','.join(x)).reset_index()
    transactionData = transactionData.drop(columns=['Member_number', 'Date'])
    transactionData.columns = ['itemDescription']

    # Process transactions
    transactions = transactionData['itemDescription'].str.split(',')

    # Encode transactions
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_array, columns=te.columns_)

    return transaction_df, te.columns_, transactions

# Load data
transaction_df, unique_items, transactions = load_data()

# Generate rules with caching
@st.cache_data
def generate_rules(transaction_df):
    """
    Generate association rules using Apriori and FP-Growth algorithms
    """
    # Apriori algorithm
    frequent_itemsets = apriori(
        transaction_df, 
        min_support=0.001, 
        use_colnames=True, 
        low_memory=True, 
        max_len=10
    )
    
    # Add num_itemsets argument (total number of transactions)
    rules_apriori = association_rules(
        frequent_itemsets, 
        num_itemsets=len(transaction_df),  # Fix: add number of transactions
        metric='confidence', 
        min_threshold=0.5
    )

    # FP-Growth algorithm
    freq_itemsets_fp = fpgrowth(
        transaction_df, 
        min_support=0.001, 
        use_colnames=True, 
        max_len=10
    )
    
    # Add num_itemsets argument
    rules_fp = association_rules(
        freq_itemsets_fp, 
        num_itemsets=len(transaction_df),  # Fix: add number of transactions
        metric='confidence', 
        min_threshold=0.5
    )

    return rules_apriori, rules_fp

# Generate rules
rules_apriori, rules_fp = generate_rules(transaction_df)

def make_prediction(antecedent, rules, top_n=5):
    """
    Generate product recommendations based on association rules
    """
    matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(antecedent))]
    
    if matching_rules.empty:
        return []
    
    top_rules = matching_rules.sort_values(by='confidence', ascending=False).head(top_n)
    predictions = top_rules['consequents'].tolist()
    
    # Format predictions and add confidence and lift information
    formatted_predictions = []
    for i, (consequent, confidence, lift) in enumerate(zip(
        top_rules['consequents'], 
        top_rules['confidence'], 
        top_rules['lift']
    ), 1):
        formatted_predictions.append({
            'rank': i,
            'product': ', '.join(list(consequent)),
            'confidence': f"{confidence:.2%}",
            'lift': f"{lift:.2f}"
        })
    
    return formatted_predictions

def main():
    st.title("🛒 Grocery Recommendation System")
    
    # Sidebar for configuration
    st.sidebar.header("🔍 Recommendation Settings")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Recommendation Algorithm",
        ["Apriori", "FP-Growth"]
    )
    
    # Product selection with search functionality
    selected_products = st.sidebar.multiselect(
        "Select Products (Up to 3)", 
        sorted(unique_items), 
        max_selections=3,
        placeholder="Choose up to 3 products"
    )
    
    # Recommendation button
    if st.sidebar.button("Get Recommendations"):
        if not selected_products:
            st.warning("Please select at least one product.")
            return
        
        # Choose rules based on selected algorithm
        current_rules = rules_apriori if algorithm == "Apriori" else rules_fp
        
        # Get recommendations
        recommendations = make_prediction(set(selected_products), current_rules)
        
        # Display recommendations
        st.header(f"🏷️ Recommendations using {algorithm}")
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.table(rec_df)
        else:
            st.warning("No recommendations found for the selected products.")
    
    # Data Visualization Section
    st.header("📊 Data Insights")
    
    # Item Frequency Visualization
    st.subheader("Top 20 Most Frequent Items")
    plt.figure(figsize=(12, 6))
    item_frequencies = transaction_df.sum().sort_values(ascending=False).head(20)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    item_frequencies.plot(kind='bar', ax=ax)
    plt.title("Top 20 Most Frequent Items in Grocery Dataset")
    plt.xlabel("Items")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()
