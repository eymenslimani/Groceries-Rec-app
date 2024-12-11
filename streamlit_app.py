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
    
    Returns:
    - transaction_df: DataFrame with binary encoded transactions
    - unique_items: List of unique items in the dataset
    - num_transactions: Total number of transactions
    """
    # Load the dataset
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

    return transaction_df, te.columns_, len(transaction_df)

# Load data
transaction_df, unique_items, num_transactions = load_data()

# Generate rules
@st.cache_data
def generate_rules(transaction_df, algorithm='apriori', min_support=0.001, min_confidence=0.5, max_len=10):
    """
    Generate association rules using Apriori or FP-Growth algorithms
    
    Args:
    - transaction_df: Binary encoded transaction DataFrame
    - algorithm: 'apriori' or 'fpgrowth'
    - min_support: Minimum support threshold
    - min_confidence: Minimum confidence threshold
    - max_len: Maximum length of itemsets
    
    Returns:
    - Generated association rules
    """
    if algorithm == 'apriori':
        frequent_itemsets = apriori(transaction_df, min_support=min_support, 
                                    use_colnames=True, low_memory=True, max_len=max_len)
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
    else:  # fp-growth
        freq_itemsets = fpgrowth(transaction_df, min_support=min_support, 
                                 use_colnames=True, max_len=max_len)
        rules = association_rules(freq_itemsets, metric='confidence', min_threshold=min_confidence)
    
    return rules

# Prediction function
def make_prediction(antecedent, rules, top_n=5):
    """
    Generate product recommendations based on input products
    
    Args:
    - antecedent: Set of input products
    - rules: Association rules DataFrame
    - top_n: Number of top recommendations
    
    Returns:
    - List of recommended products
    """
    matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(antecedent))]
    top_rules = matching_rules.sort_values(by='confidence', ascending=False).head(top_n)
    
    predictions = []
    for consequent in top_rules['consequents']:
        pred_items = ', '.join(list(consequent))
        confidence = top_rules[top_rules['consequents'] == consequent]['confidence'].values[0]
        predictions.append((pred_items, confidence))
    
    return predictions

# Main Streamlit App
def main():
    st.title("🛒 Grocery Recommendation App")
    st.markdown("Find product recommendations based on association rule mining!")

    # Sidebar for settings
    st.sidebar.header("Recommendation Settings")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Recommendation Algorithm",
        ["Apriori", "FP-Growth"]
    )

    # Support and confidence sliders
    min_support = st.sidebar.slider(
        "Minimum Support", 
        min_value=0.0001, 
        max_value=0.1, 
        value=0.001, 
        step=0.0005
    )
    
    min_confidence = st.sidebar.slider(
        "Minimum Confidence", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1
    )

    # Multi-select for products
    selected_products = st.sidebar.multiselect(
        "Select Products", 
        sorted(unique_items), 
        max_selections=3,
        placeholder="Choose up to 3 products"
    )

    # Generate rules based on selected algorithm and thresholds
    current_rules = generate_rules(
        transaction_df, 
        algorithm.lower(), 
        min_support=min_support, 
        min_confidence=min_confidence
    )

    # Recommendation Section
    if st.sidebar.button("Get Recommendations"):
        if not selected_products:
            st.warning("Please select at least one product.")
        else:
            # Get predictions
            recommendations = make_prediction(set(selected_products), current_rules)

            # Display results
            st.subheader(f"Recommendations using {algorithm}")
            st.markdown(f"**Selected Products:** {', '.join(selected_products)}")
            
            if recommendations:
                st.success("Recommended Products:")
                for rec, conf in recommendations:
                    st.info(f"➡️ {rec} (Confidence: {conf:.2%})")
            else:
                st.warning("No recommendations found for the selected products.")

    # Data Insights Section
    st.header("Data Insights")

    # Item Frequency Plot
    st.subheader("Top 20 Most Frequent Items")
    plt.figure(figsize=(12, 6))
    item_frequencies = transaction_df.sum().sort_values(ascending=False).head(20)
    sns.barplot(x=item_frequencies.index, y=item_frequencies.values)
    plt.title("Top 20 Most Frequent Items")
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt)

    # Top Rules Heatmap
    st.subheader("Top 10 Association Rules Heatmap")
    top10_rules = current_rules.nlargest(10, 'confidence')
    plt.figure(figsize=(10, 6))
    sns.heatmap(top10_rules[['support', 'confidence', 'lift']], 
                annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Top 10 Rules: Support, Confidence, Lift")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
