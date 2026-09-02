import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Set page title and icon
st.set_page_config(page_title="🛒 Product Recommender", page_icon="📊")

# Load data and model
@st.cache_resource
def load_data_and_model():
    with open('artifacts/final_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    model = tf.keras.models.load_model('artifacts/rnn_recommender_optimized.h5')
    with open('artifacts/item_tokenizer_final.pkl', 'rb') as f:
        item_tokenizer = pickle.load(f)
    with open('artifacts/category_tokenizer_final.pkl', 'rb') as f:
        category_tokenizer = pickle.load(f)
    return data, model, item_tokenizer, category_tokenizer

# Cache sampled visitor IDs
@st.cache_data
def cache_sampled_visitors(_data):
    visitor_counts = _data.groupby('visitorid').size()
    valid_visitors = visitor_counts[(visitor_counts >= 3) & (visitor_counts <= 28)].index.tolist()
    return np.random.choice(valid_visitors, size=100, replace=False).tolist()

# Load data
data_full, model, item_tokenizer, category_tokenizer = load_data_and_model()

# Ensure data_full['itemid'] is integer
if data_full['itemid'].dtype == 'object':
    data_full['itemid'] = pd.to_numeric(data_full['itemid'], errors='coerce').fillna(0).astype(int)

# Get sampled visitor IDs and filter data
sampled_visitors = cache_sampled_visitors(data_full)
data = data_full[data_full['visitorid'].isin(sampled_visitors)]

# App header
st.title("🛒 Product Recommendation System 📊")
st.markdown(
    "Welcome to our e-commerce recommender! Select a visitor ID or enter an item sequence to get personalized product suggestions. 🚀")

# Dropdown menu for naive users
with st.expander("ℹ️ How This App Helps You"):
    st.markdown("""
    **What is this app?**  
    This app suggests products you might like based on your past shopping actions (like adding items to your cart or buying them). It uses a smart model (LSTM) trained on Retailrocket data to predict what you’ll want next. 🧠

    **How does it help?**  
    - **Personalized Suggestions**: Find products tailored to your interests. 🎯  
    - **Save Time**: Quickly discover items you’re likely to buy. ⏰  
    - **Explore Categories**: See product categories to make better choices. 🛍️  

    **How to use it?**  
    1. Choose a **Visitor ID** from the dropdown (randomly sampled users with 3–28 items and valid categories) or search for one.  
    2. Or, enter a **sequence of item IDs** (e.g., '355908 248676') to get custom predictions.  
    3. Click **Get Recommendations** to view the top-5 suggested products with their categories and confidence scores. ✅  
    **Note**: Confidence scores are normalized for display. Low raw scores are normal due to 16,435 possible items. 📊
    """)

# Input section
st.header("🔍 Get Your Recommendations")
input_type = st.radio("Choose input method:", ("Select Visitor ID", "Search Visitor ID", "Enter Item Sequence"))

if input_type == "Select Visitor ID":
    selected_visitor = st.selectbox("Select a Visitor ID:", data['visitorid'].unique().tolist())
    if selected_visitor:
        user_data = data[data['visitorid'] == selected_visitor]
        item_sequence = user_data['itemid'].astype(str).tolist()
        category_sequence = user_data['value'].astype(str).tolist()
        st.write(f"**Your Sequence**: {', '.join(item_sequence)}")
        st.write(f"**Categories**: {', '.join(category_sequence)}")
elif input_type == "Search Visitor ID":
    search_term = st.text_input("Search for a Visitor ID:")
    if search_term:
        filtered_visitors = [vid for vid in data['visitorid'].unique() if str(vid).startswith(search_term)]
        if filtered_visitors:
            selected_visitor = st.selectbox("Select a matching Visitor ID:", filtered_visitors)
            user_data = data[data['visitorid'] == selected_visitor]
            item_sequence = user_data['itemid'].astype(str).tolist()
            category_sequence = user_data['value'].astype(str).tolist()
            st.write(f"**Your Sequence**: {', '.join(item_sequence)}")
            st.write(f"**Categories**: {', '.join(category_sequence)}")
        else:
            st.error("No matching Visitor ID found. Try another search term.")
            item_sequence = []
            category_sequence = []
    else:
        item_sequence = []
        category_sequence = []
else:
    item_sequence_input = st.text_input("Enter item IDs (space-separated, e.g., '355908 248676'):")
    item_sequence = item_sequence_input.strip().split() if item_sequence_input else []
    category_sequence = []
    if item_sequence:
        for item in item_sequence:
            try:
                item_id = int(item)
                if item_id in data['itemid'].values:
                    cat = data[data['itemid'] == item_id]['value'].iloc[0]
                    category_sequence.append(str(cat))
                else:
                    category_sequence.append('0')
            except ValueError:
                category_sequence.append('0')  # Fallback for invalid input
        st.write(f"**Your Sequence**: {', '.join(item_sequence)}")
        st.write(f"**Categories**: {', '.join(category_sequence)}")

# Prediction function
def get_recommendations(item_seq, cat_seq, max_sequence_length=28):
    # Convert Series to lists if necessary
    if isinstance(item_seq, pd.Series):
        item_seq = item_seq.tolist()
    if isinstance(cat_seq, pd.Series):
        cat_seq = cat_seq.tolist()

    if not item_seq or len(item_seq) != len(cat_seq):
        return None, "Please provide a valid item sequence with corresponding categories."

    # Validate item IDs
    valid_items = [item for item in item_seq if item in item_tokenizer.word_index]
    if not valid_items:
        return None, "No valid item IDs found. Please check your input."

    # Tokenize and pad sequences
    item_seq_tokens = item_tokenizer.texts_to_sequences([item_seq])
    item_padded = pad_sequences(item_seq_tokens, maxlen=max_sequence_length, padding='pre')
    cat_seq_tokens = category_tokenizer.texts_to_sequences([cat_seq])
    cat_padded = pad_sequences(cat_seq_tokens, maxlen=max_sequence_length, padding='pre')

    # Predict
    item_padded = np.array(item_padded)
    cat_padded = np.array(cat_padded)
    pred_probs = model.predict([item_padded, cat_padded], verbose=0)

    # Get top-5 recommendations
    top_5_indices = np.argsort(pred_probs[0])[-5:][::-1]
    top_5_probs = pred_probs[0][top_5_indices]

    # Normalize confidence scores
    top_5_probs = top_5_probs / top_5_probs.sum() if top_5_probs.sum() > 0 else top_5_probs

    # Map item IDs back to original IDs
    reverse_item_map = {v: k for k, v in item_tokenizer.word_index.items()}
    recommended_items = [reverse_item_map.get(idx, '0') for idx in top_5_indices]

    # Get categories using full dataset
    categories = []
    for item in recommended_items:
        try:
            item_id = int(item)
            if item_id in data_full['itemid'].values:
                cat = data_full[data_full['itemid'] == item_id]['value'].iloc[0]
                categories.append(str(cat))
            else:
                categories.append('Unknown')
        except ValueError:
            categories.append('Unknown')

    # Create descriptive item names
    item_names = [f"Item {item}" if item != '0' else 'Unknown Product' for item in recommended_items]

    return pd.DataFrame({
        'Product': item_names,
        'Item ID': recommended_items,
        'Category ID': categories,
        'Confidence': [f"{p:.4f}" for p in top_5_probs]
    }), None

# Get recommendations button
if st.button("Get Recommendations 🎉"):
    if (input_type in ["Select Visitor ID",
                       "Search Visitor ID"] and 'selected_visitor' in locals() and selected_visitor) or (
            input_type == "Enter Item Sequence" and item_sequence):
        recommendations, error = get_recommendations(item_sequence, category_sequence)
    else:
        recommendations, error = None, "Please select a Visitor ID or enter an item sequence."

    if error:
        st.error(error)
    else:
        st.subheader("🎁 Top-5 Recommended Products")
        st.dataframe(recommendations)

        # Visualize confidence scores
        st.subheader("📈 Confidence Scores")
        item_ids = recommendations['Item ID'].values
        confidences = recommendations['Confidence'].astype(float).values

        # Create bar plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(item_ids, confidences, color='skyblue')
        ax.set_xlabel('Item ID')
        ax.set_ylabel('Confidence Score (Normalized)')
        ax.set_title('Top-5 Recommendations Confidence')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)