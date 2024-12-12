import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter

# Page Configuration
st.set_page_config(page_title="Advanced Grocery Recommendation App", page_icon="🛒", layout="wide")

# Enhanced data loading with more preprocessing
@st.cache_data
def load_data():
    """
    Load and preprocess the grocery dataset with enhanced preprocessing
    """
    try:
        # Load the dataset from GitHub raw file
        data = pd.read_csv('https://raw.githubusercontent.com/eymenslimani/data/refs/heads/main/Groceries_dataset.csv')
        
        # More robust preprocessing
        data.dropna(inplace=True)

        # Aggregate transactions by Member and Date
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
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Enhanced rules generation with better parameters
@st.cache_data
def generate_rules(transaction_df):
    """
    Generate association rules with more sophisticated parameters
    """
    try:
        num_transactions = len(transaction_df)

        # Apriori with more nuanced parameters
        frequent_itemsets_apriori = apriori(
            transaction_df, 
            min_support=0.05,  # Increased minimum support 
            use_colnames=True, 
            low_memory=True, 
            max_len=5  # Limit to more focused itemsets
        )
        
        rules_apriori = association_rules(
            frequent_itemsets_apriori, 
            metric='lift',  # Added lift as primary metric
            min_threshold=1.5,  # More stringent lift threshold
            num_itemsets=num_transactions
        )

        # FP-Growth with similar improvements
        frequent_itemsets_fp = fpgrowth(
            transaction_df, 
            min_support=0.05,  # Matching Apriori's support
            use_colnames=True, 
            max_len=5
        )
        
        rules_fp = association_rules(
            frequent_itemsets_fp, 
            metric='lift',
            min_threshold=1.5,
            num_itemsets=num_transactions
        )

        return rules_apriori, rules_fp
    except Exception as e:
        st.error(f"Error generating rules: {e}")
        return None, None

def make_prediction(antecedent, rules, top_n=5):
    """
    Enhanced recommendation generation with more sophisticated matching
    """
    try:
        antecedent = frozenset(antecedent) if not isinstance(antecedent, frozenset) else antecedent
        
        # More intelligent rule matching
        matching_rules = rules[
            rules['antecedents'].apply(
                lambda x: x.issubset(antecedent) or 
                len(x.intersection(antecedent)) > 0
            )
        ]
        
        if matching_rules.empty:
            return []
        
        # Prioritize rules with higher lift and confidence
        top_rules = matching_rules.sort_values(
            by=['lift', 'confidence'], 
            ascending=False
        ).head(top_n)
        
        # More detailed recommendation formatting
        formatted_predictions = []
        for i, (consequent, lift, confidence, support) in enumerate(zip(
            top_rules['consequents'], 
            top_rules['lift'],
            top_rules['confidence'],
            top_rules['support']
        ), 1):
            formatted_predictions.append({
                'rank': i,
                'product': ', '.join(list(consequent)),
                'lift': f"{lift:.2f}",
                'confidence': f"{confidence:.2%}",
                'support': f"{support:.4f}"
            })
        
        return formatted_predictions
    except Exception as e:
        st.error(f"Error in recommendation generation: {e}")
        return []

def visualize_item_frequencies(transaction_df, top_n=20):
    """
    Enhanced item frequency visualization
    """
    item_frequencies = transaction_df.sum().sort_values(ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.barplot(x=item_frequencies.index, y=item_frequencies.values, palette='viridis')
    plt.title(f"Top {top_n} Most Frequent Items in Grocery Dataset", fontsize=15)
    plt.xlabel("Items", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    
    return fig

def main():
    st.title("🛒 Advanced Grocery Recommendation System")
    
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
    
    st.subheader(f"Top 20 Most Frequent Items")
    fig = visualize_item_frequencies(transaction_df)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
