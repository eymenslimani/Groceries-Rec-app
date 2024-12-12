import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter

# Page Configuration
st.set_page_config(page_title="Grocery Recommendation App", page_icon="🛒", layout="wide")

# Enhanced data loading with more preprocessing
@st.cache_data
def load_data():
    """
    Load and preprocess the grocery dataset with robust preprocessing
    """
    try:
        # Load the dataset 
        data = pd.read_csv('https://raw.githubusercontent.com/eymenslimani/data/refs/heads/main/Groceries_dataset.csv')
        
        # Basic preprocessing
        data.dropna(subset=['itemDescription'], inplace=True)

        # Aggregate transactions by Member and Date
        transaction_data = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()
        
        # Prepare unique transactions
        transactions = [list(set(trans)) for trans in transaction_data['itemDescription']]

        # Encode transactions
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        transaction_df = pd.DataFrame(te_array, columns=te.columns_)

        return transaction_df, te.columns_, transactions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Simplified rules generation with error handling
@st.cache_data
def generate_rules(transaction_df):
    """
    Generate association rules with refined parameters
    """
    try:
        # Total number of transactions
        num_transactions = len(transaction_df)

        # Generate Apriori Rules
        frequent_itemsets_apriori = apriori(
            transaction_df, 
            min_support=0.01,  # Adjusted support threshold
            use_colnames=True, 
            max_len=3  # Limit itemset length
        )
        
        rules_apriori = association_rules(
            frequent_itemsets_apriori, 
            metric='confidence',
            min_threshold=0.2,  # Adjusted confidence threshold
            num_itemsets=num_transactions
        )

        # Generate FP-Growth Rules
        frequent_itemsets_fp = fpgrowth(
            transaction_df, 
            min_support=0.01,  
            use_colnames=True, 
            max_len=3
        )
        
        rules_fp = association_rules(
            frequent_itemsets_fp, 
            metric='confidence',
            min_threshold=0.2,
            num_itemsets=num_transactions
        )

        return rules_apriori, rules_fp
    except Exception as e:
        st.error(f"Error generating rules: {e}")
        return None, None

def make_prediction(antecedent, rules, top_n=5):
    """
    Generate recommendations based on input items
    """
    try:
        # Convert antecedent to a frozenset
        antecedent = frozenset(antecedent) if not isinstance(antecedent, frozenset) else antecedent
        
        # Find matching rules
        matching_rules = rules[
            rules['antecedents'].apply(
                lambda x: len(x.intersection(antecedent)) > 0
            )
        ]
        
        if matching_rules.empty:
            return []
        
        # Sort and select top recommendations
        top_rules = matching_rules.sort_values(
            by=['confidence', 'lift'], 
            ascending=False
        ).head(top_n)
        
        # Format predictions
        formatted_predictions = []
        for i, (consequent, confidence, lift, support) in enumerate(zip(
            top_rules['consequents'], 
            top_rules['confidence'],
            top_rules['lift'],
            top_rules['support']
        ), 1):
            formatted_predictions.append({
                'rank': i,
                'product': ', '.join(list(consequent)),
                'confidence': f"{confidence:.2%}",
                'lift': f"{lift:.2f}",
                'support': f"{support:.4f}"
            })
        
        return formatted_predictions
    except Exception as e:
        st.error(f"Error in recommendation generation: {e}")
        return []

def visualize_item_frequencies(transaction_df, top_n=20):
    """
    Create item frequency visualization
    """
    item_frequencies = transaction_df.sum().sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(15, 7))
    sns.barplot(x=item_frequencies.index, y=item_frequencies.values, palette='viridis')
    plt.title(f"Top {top_n} Most Frequent Items", fontsize=15)
    plt.xlabel("Items", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    
    return plt.gcf()

def visualize_top_rules(rules, top_n=10):
    """
    Visualize top association rules as a network
    """
    top_rules = rules.nlargest(top_n, 'lift')
    
    G = nx.DiGraph()
    for _, rule in top_rules.iterrows():
        for antecedent in rule['antecedents']:
            for consequent in rule['consequents']:
                G.add_edge(antecedent, consequent, weight=rule['lift'])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]

    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            font_size=8, node_size=1500, 
            edge_color=weights, edge_cmap=plt.cm.viridis, 
            width=2)
    
    plt.title("Association Rules Network", fontsize=15)
    return plt.gcf()

def main():
    st.title("🛒 Grocery Recommendation System")
    
    # Load data
    transaction_df, unique_items, transactions = load_data()
    
    if transaction_df is None:
        st.error("Failed to load data. Please check your connection.")
        return
    
    # Generate rules
    rules_apriori, rules_fp = generate_rules(transaction_df)
    
    if rules_apriori is None or rules_fp is None:
        st.error("Failed to generate association rules.")
        return

    # Sidebar configuration
    st.sidebar.header("🔍 Recommendation Settings")
    
    algorithm = st.sidebar.selectbox(
        "Select Recommendation Algorithm",
        ["Apriori", "FP-Growth"]
    )
    
    selected_products = st.sidebar.multiselect(
        "Select Products (Up to 3)", 
        sorted(unique_items), 
        max_selections=3,
        placeholder="Choose up to 3 products"
    )
    
    if st.sidebar.button("Get Recommendations"):
        if not selected_products:
            st.warning("Please select at least one product.")
            return
        
        current_rules = rules_apriori if algorithm == "Apriori" else rules_fp
        
        recommendations = make_prediction(set(selected_products), current_rules)
        
        st.header(f"🏷️ Recommendations using {algorithm}")
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.table(rec_df)
        else:
            st.warning(f"No recommendations found for {', '.join(selected_products)}. Try different products.")
    
    # Data Visualization Section
    st.header("📊 Data Insights")
    
    # Item Frequencies
    st.subheader("Top 20 Most Frequent Items")
    fig_frequencies = visualize_item_frequencies(transaction_df)
    st.pyplot(fig_frequencies)
    plt.close(fig_frequencies)
    
    # Rules Visualization
    st.subheader("Top Association Rules Network")
    current_rules = rules_apriori  # You can toggle between apriori and fp-growth
    fig_rules = visualize_top_rules(current_rules)
    st.pyplot(fig_rules)
    plt.close(fig_rules)

if __name__ == "__main__":
    main()
