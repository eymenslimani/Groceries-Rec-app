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
def generate_rules(transaction_df, num_transactions):
    try:
        # Apriori algorithm
        frequent_itemsets = apriori(transaction_df, min_support=0.001, use_colnames=True, low_memory=True, max_len=10)
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5, num_itemsets=num_transactions) 

        # FP-Growth algorithm
        freq_itemsets_fp = fpgrowth(transaction_df, min_support=0.001, use_colnames=True, max_len=10)
        rules_fp = association_rules(freq_itemsets_fp, metric='confidence', min_threshold=0.5, num_itemsets=num_transactions) 

        return rules, rules_fp
    except Exception as e:
        st.error(f"Error generating rules: {e}")
        return None, None  # Return None gracefully

# Generate rules
rules, rules_fp = generate_rules(transaction_df, num_transactions)

# Handle potential errors in rule generation
if rules is None or rules_fp is None:
    st.error("Error generating association rules. Please check the logs for details.")
    st.stop()  # Stop execution if rule generation fails

# Ensure numeric columns
rules['confidence'] = pd.to_numeric(rules['confidence'], errors='coerce')
rules = rules.dropna(subset=['confidence'])
rules_fp['confidence'] = pd.to_numeric(rules_fp['confidence'], errors='coerce')
rules_fp = rules_fp.dropna(subset=['confidence'])

# Prediction function
def make_prediction(antecedent, rules, top_n=5):
    matching_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(antecedent))]
    top_rules = matching_rules.sort_values(by='confidence', ascending=False).head(top_n)
    predictions = top_rules['consequents'].tolist()
    formatted_predictions = [', '.join(list(consequent)) for consequent in predictions]
    return formatted_predictions

# Streamlit App
def main():
    st.title("🛒 Grocery Recommendation App")

    # Sidebar for algorithm selection and product input
    st.sidebar.header("Recommendation Settings")
    algorithm = st.sidebar.selectbox(
        "Select Recommendation Algorithm",
        ["Apriori", "FP-Growth"]
    )

    # Multi-select for products
    selected_products = st.sidebar.multiselect(
        "Select Products",
        sorted(unique_items),
        max_selections=3,
        placeholder="Choose up to 3 products"
    )

    # Recommendation Section
    if st.sidebar.button("Get Recommendations"):
        if not selected_products:
            st.warning("Please select at least one product.")
            return

        # Choose rules based on algorithm
        current_rules = rules if algorithm == "Apriori" else rules_fp

        # Get predictions
        recommendations = make_prediction(set(selected_products), current_rules)

        # Display results
        st.subheader(f"Recommendations using {algorithm}")
        if recommendations:
            st.success("Recommended Products:")
            for rec in recommendations:
                st.info(f"➡️ {rec}")
        else:
            st.warning("No recommendations found for the selected products.")

    # Data Visualization Section
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
    if not current_rules.empty:  # Check if current_rules has data before plotting
        top10_rules = current_rules.nlargest(10, 'confidence')
        plt.figure(figsize=(10, 6))
        sns.heatmap(top10_rules[['support', 'confidence', 'lift']],
                     annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Top 10 Rules: Support, Confidence, Lift")
        st.pyplot(plt)

if __name__ == "__main__":
    main()
