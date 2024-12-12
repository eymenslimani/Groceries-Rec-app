import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter

# Page Configuration
st.set_page_config(page_title="Grocery Recommendation App", page_icon="🛒", layout="wide")

# Caching data loading for performance
@st.cache_data
def load_data():
    """
    Load and preprocess the grocery dataset with more detailed processing
    """
    try:
        # Load the dataset 
        data = pd.read_csv('Groceries_dataset.csv')
        data.dropna(inplace=True)

        # Aggregating Transactions by Member and Date
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

# Generate rules with caching and improved parameters
@st.cache_data
def generate_rules(transaction_df):
    """
    Generate association rules using Apriori and FP-Growth algorithms with refined parameters
    """
    try:
        # Total number of transactions
        num_transactions = len(transaction_df)

        # Apriori algorithm with refined parameters
        frequent_itemsets_apriori = apriori(
            transaction_df, 
            min_support=0.3,  # Increased support threshold 
            use_colnames=True, 
            low_memory=True, 
            max_len=10
        )
        
        rules_apriori = association_rules(
            frequent_itemsets_apriori, 
            metric='confidence', 
            min_threshold=0.7,  # Increased confidence threshold
            num_itemsets=num_transactions
        )

        # FP-Growth algorithm with similar refinements
        frequent_itemsets_fp = fpgrowth(
            transaction_df, 
            min_support=0.3,  # Increased support threshold
            use_colnames=True, 
            max_len=10
        )
        
        rules_fp = association_rules(
            frequent_itemsets_fp, 
            metric='confidence', 
            min_threshold=0.7,  # Increased confidence threshold
            num_itemsets=num_transactions
        )

        return rules_apriori, rules_fp
    except Exception as e:
        st.error(f"Error generating rules: {e}")
        return None, None

def make_prediction(antecedent, rules, top_n=3):
    """
    Generate product recommendations with improved prediction logic
    """
    try:
        # Convert antecedent to frozenset if it's not already
        antecedent = frozenset(antecedent) if not isinstance(antecedent, frozenset) else antecedent
        
        # More sophisticated matching strategy
        matching_rules = rules[
            rules['antecedents'].apply(
                lambda x: x.issubset(antecedent) or 
                len(x.intersection(antecedent)) > 0
            )
        ]
        
        if matching_rules.empty:
            return []
        
        # Sort rules by lift and confidence
        top_rules = matching_rules.sort_values(
            by=['lift', 'confidence'], 
            ascending=False
        ).head(top_n)
        
        # Enhanced prediction formatting
        unique_predictions = set()
        formatted_predictions = []
        
        for i, (consequent, lift, confidence, support) in enumerate(zip(
            top_rules['consequents'], 
            top_rules['lift'],
            top_rules['confidence'],
            top_rules['support']
        ), 1):
            for item in consequent:
                if item not in unique_predictions:
                    unique_predictions.add(item)
                    formatted_predictions.append({
                        'rank': i,
                        'product': item,
                        'lift': f"{lift:.2f}",
                        'confidence': f"{confidence:.2%}",
                        'support': f"{support:.4f}"
                    })
                    
                    if len(formatted_predictions) == top_n:
                        break
            
            if len(formatted_predictions) == top_n:
                break
        
        return formatted_predictions
    except Exception as e:
        st.error(f"Error in recommendation generation: {e}")
        return []

def main():
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
            st.warning(f"No recommendations found for {', '.join(selected_products)}. Try different products.")
    
    # Data Visualization Section
    st.header("📊 Data Insights")
    
    # Item Frequency Visualization with more detailed plotting
    st.subheader("Top 20 Most Frequent Items")
    plt.figure(figsize=(12, 6))
    
    # Improved item frequency calculation
    item_frequencies = transaction_df.sum().sort_values(ascending=False).head(20)
    
    # Create bar plot with color gradient
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = item_frequencies.plot(kind='bar', ax=ax, 
                                 color=plt.cm.Pastel2(np.linspace(0, 1, 20)))
    plt.title("Top 20 Most Frequent Items in Grocery Dataset")
    plt.xlabel("Items")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()
