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
        data.dropna(inplace=True)

        # Aggregate transactions by Member and Date
        transaction_data = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()
        # Dropping Unnecessary Columns: Member Number and Date
        transaction_data = transaction_data.drop(columns=['Member_number', 'Date'])
        transaction_data.columns = ['itemDescription']
       
        df = transaction_data.reset_index(drop=True)
        transactions = df['itemDescription'].tolist()
        
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
        # Generate Apriori Rules
        frequent_itemsets_apriori = apriori(transaction_df, min_support=0.3, use_colnames=True, low_memory=True, max_len=10)
        rules_apriori = association_rules(frequent_itemsets_apriori, metric='confidence', min_threshold=0.7)

        # Generate FP-Growth Rules
        frequent_itemsets_fp = fpgrowth(transaction_df, min_support=0.3, use_colnames=True, max_len=10)
        rules_fp = association_rules(frequent_itemsets_fp, metric='confidence', min_threshold=0.7)

        return rules_apriori, rules_fp
    except Exception as e:
        st.error(f"Error generating rules: {e}")
        return None, None

def make_prediction(antecedent, rules, top_n=3):
    """
    Generate recommendations based on input items
    """
    try:
        matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(antecedent))]
        
        # Select unique recommendations with highest lift
        recommendations = {}
        for _, rule in matching_rules.iterrows():
            for consequent in rule['consequents']:
                if consequent not in antecedent:
                    if consequent not in recommendations or rule['lift'] > recommendations[consequent]['Lift']:
                        recommendations[consequent] = {
                            'Lift': rule['lift'],
                            'Confidence': rule['confidence'],
                            'Support': rule['support']
                        }

        # Convert to sorted list by lift
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1]['Lift'], reverse=True)[:top_n]
        
        return [{"Item": item, **metrics} for item, metrics in sorted_recommendations]
    except Exception as e:
        st.error(f"Error in recommendation generation: {e}")
        return []

def plot_item_frequency(transactions, top_n=20):
    item_counts = Counter(item for sublist in transactions for item in sublist)
    most_common_items = item_counts.most_common(top_n)
    
    items, counts = zip(*most_common_items)
    plt.bar(items, counts, color=plt.cm.Pastel2(range(len(items))))
    plt.title("Absolute Item Frequency Plot")
    plt.xticks(rotation=90)
    plt.ylabel("Frequency")
    plt.show()

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
        "Select Products", 
        sorted(unique_items), 
        placeholder="Choose products"
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
    fig_frequencies = plt.figure()
    plot_item_frequency(transactions)
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

