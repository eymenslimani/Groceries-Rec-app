import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Streamlit App Title
st.title("Grocery Product Recommendation System")

# File Upload
uploaded_file = st.file_uploader("Upload your transaction dataset (CSV format):", type="csv")

if uploaded_file:
    # Load Dataset
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(data.head())

        # Ensure data format is correct
        st.write("### Dataset Information")
        st.write(data.info())

        # Check for proper structure
        if 'Transaction' in data.columns and 'Item' in data.columns:
            st.success("Dataset format looks good.")

            # Preprocess Data
            st.write("### Preprocessing the Data")
            transaction_df = data.groupby('Transaction')['Item'].apply(list).reset_index()
            st.write(transaction_df.head())

            # Create Basket Format for One-Hot Encoding
            st.write("### Transforming Data for Apriori Analysis")
            basket = pd.get_dummies(transaction_df['Item'].apply(pd.Series).stack()).sum(level=0)
            st.write(basket.head())

            # Apriori Algorithm
            min_support = st.slider("Select Minimum Support", min_value=0.01, max_value=0.5, value=0.01, step=0.01)
            frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

            st.write("### Frequent Itemsets")
            st.dataframe(frequent_itemsets)

            # Association Rules
            min_confidence = st.slider("Select Minimum Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

            st.write("### Association Rules")
            st.dataframe(rules)

            # Display Columns for Filtering Rules
            st.write("### Filter Rules")
            selected_metric = st.selectbox("Select Metric for Filtering", rules.columns)
            threshold = st.slider(f"Select Minimum {selected_metric}", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

            filtered_rules = rules[rules[selected_metric] >= threshold]
            st.write(f"### Filtered Rules by {selected_metric} >= {threshold}")
            st.dataframe(filtered_rules)

        else:
            st.error("Dataset must contain 'Transaction' and 'Item' columns.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV file containing transaction data.")
