import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from typing import List, Dict, Any

# Page Configuration
st.set_page_config(page_title="Smart Grocery Recommender", page_icon="🛒", layout="wide")

@st.cache_data
def load_and_preprocess_data(url: str = 'https://raw.githubusercontent.com/yourusername/your-repo/main/groceries_dataset.csv'):
    """
    Enhanced data loading with category-aware preprocessing
    """
    try:
        # Load data
        data = pd.read_csv(url)
        
        # Drop duplicates and reset index
        data.drop_duplicates(inplace=True)
        
        # Preprocess transactions
        transactions_by_member = data.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()
        
        # Create transaction encoder
        te = TransactionEncoder()
        te_array = te.fit(transactions_by_member['itemDescription']).transform(transactions_by_member['itemDescription'])
        transaction_df = pd.DataFrame(te_array, columns=te.columns_)
        
        # Additional metadata
        category_mapping = data.groupby('itemDescription')['Category'].first()
        
        return {
            'transaction_df': transaction_df, 
            'unique_items': te.columns_,
            'category_mapping': category_mapping
        }
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None

@st.cache_data
def generate_association_rules(transaction_df: pd.DataFrame):
    """
    Generate rules with category-aware parameters
    """
    try:
        # Apriori Rules
        frequent_itemsets_apriori = apriori(
            transaction_df, 
            min_support=0.02,  # Adjusted for potentially smaller dataset
            use_colnames=True, 
            max_len=3
        )
        
        apriori_rules = association_rules(
            frequent_itemsets_apriori, 
            metric='lift', 
            min_threshold=1.2
        )
        
        # FP-Growth Rules
        frequent_itemsets_fp = fpgrowth(
            transaction_df, 
            min_support=0.02,
            use_colnames=True, 
            max_len=3
        )
        
        fp_rules = association_rules(
            frequent_itemsets_fp, 
            metric='lift', 
            min_threshold=1.2
        )
        
        return apriori_rules, fp_rules
    
    except Exception as e:
        st.error(f"Rule generation error: {e}")
        return None, None

def smart_recommendations(
    antecedent: set, 
    rules: pd.DataFrame, 
    category_mapping: pd.Series, 
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Enhanced recommendation system with category insights
    """
    try:
        antecedent = frozenset(antecedent)
        
        # Intelligent rule matching
        matching_rules = rules[
            rules['antecedents'].apply(lambda x: len(x.intersection(antecedent)) > 0)
        ]
        
        if matching_rules.empty:
            return []
        
        # Sort and select top recommendations
        top_recommendations = matching_rules.sort_values(
            by=['lift', 'confidence'], 
            ascending=False
        ).head(top_n)
        
        recommendations = []
        for i, (consequent, lift, confidence, support) in enumerate(zip(
            top_recommendations['consequents'],
            top_recommendations['lift'],
            top_recommendations['confidence'],
            top_recommendations['support']
        ), 1):
            product = list(consequent)[0]
            recommendations.append({
                'rank': i,
                'product': product,
                'category': category_mapping.get(product, 'Unknown'),
                'lift': f"{lift:.2f}",
                'confidence': f"{confidence:.2%}",
                'support': f"{support:.4f}"
            })
        
        return recommendations
    
    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return []

def main():
    st.title("🛒 Smart Grocery Recommendation System")
    
    # Load preprocessed data
    data_dict = load_and_preprocess_data()
    
    if not data_dict:
        st.error("Data loading failed. Check connection.")
        return
    
    # Unpack data
    transaction_df = data_dict['transaction_df']
    unique_items = data_dict['unique_items']
    category_mapping = data_dict['category_mapping']
    
    # Generate rules
    apriori_rules, fp_rules = generate_association_rules(transaction_df)
    
    if apriori_rules is None or fp_rules is None:
        st.error("Rule generation failed.")
        return
    
    # Sidebar Configuration
    st.sidebar.header("🔍 Recommendation Settings")
    
    algorithm = st.sidebar.selectbox(
        "Select Algorithm", 
        ["Apriori", "FP-Growth"]
    )
    
    selected_products = st.sidebar.multiselect(
        "Select Products (Max 3)", 
        sorted(unique_items), 
        max_selections=3
    )
    
    # Recommendation Generation
    if st.sidebar.button("Get Recommendations"):
        if not selected_products:
            st.warning("Select at least one product.")
            return
        
        current_rules = apriori_rules if algorithm == "Apriori" else fp_rules
        
        recommendations = smart_recommendations(
            set(selected_products), 
            current_rules, 
            category_mapping
        )
        
        st.header(f"🏷️ Recommendations ({algorithm})")
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.table(rec_df)
        else:
            st.warning("No recommendations found.")
    
    # Category Distribution Visualization
    st.header("📊 Category Insights")
    category_counts = category_mapping.value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title("Product Category Distribution")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
