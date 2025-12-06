import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Config & Load
# ---------------------------------------------------------
st.set_page_config(page_title="Customer Segmentation AI", page_icon="🧩", layout="wide")

@st.cache_resource
def load_data():
    try:
        return joblib.load('segmentation_artifacts.joblib')
    except FileNotFoundError:
        return None

data = load_data()

if not data:
    st.error("⚠️ Artifacts not found! Run `train_segmentation.py` first.")
    st.stop()

scaler = data['scaler']
kmeans = data['kmeans_model']
features = data['features']
cluster_mapping = data['cluster_mapping']
cluster_means = data['means']

# ---------------------------------------------------------
# 2. Sidebar Input
# ---------------------------------------------------------
st.sidebar.header("📝 Customer Profile")
st.sidebar.markdown("Adjust the values to see the persona change.")

total_amount = st.sidebar.number_input("Total Spend ($)", 0.0, 20000.0, 500.0, step=50.0)
quantity = st.sidebar.slider("Order Quantity", 1, 20, 3)
discount = st.sidebar.number_input("Discount Used ($)", 0.0, 5000.0, 0.0, step=10.0)
duration = st.sidebar.slider("Session Duration (min)", 1, 60, 15)
pages = st.sidebar.slider("Pages Viewed", 1, 30, 8)
rating = st.sidebar.slider("Customer Rating (1-5)", 1, 5, 4)

# ---------------------------------------------------------
# 3. Main Prediction Logic
# ---------------------------------------------------------
# Prepare input vector
input_df = pd.DataFrame([{
    'Total_Amount': total_amount,
    'Quantity': quantity,
    'Discount_Amount': discount,
    'Session_Duration_Minutes': duration,
    'Pages_Viewed': pages,
    'Customer_Rating': rating
}])

# Scale & Predict
input_scaled = scaler.transform(input_df)
predicted_cluster = kmeans.predict(input_scaled)[0]
persona = cluster_mapping[predicted_cluster]

# ---------------------------------------------------------
# 4. Dashboard UI
# ---------------------------------------------------------
st.title("🧩 Customer Segmentation Engine")
st.markdown("Identify customer personas using Unsupervised Learning (K-Means).")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Result")
    # Dynamic styling based on persona
    if "Big Spender" in persona['name']:
        st.success(f"### {persona['name']}")
    elif "At-Risk" in persona['name']:
        st.error(f"### {persona['name']}")
    else:
        st.info(f"### {persona['name']}")
        
    st.markdown(f"**Description:** {persona['desc']}")
    st.markdown(f"**🚀 Recommended Action:**\n\n> {persona['action']}")

with col2:
    st.subheader("📍 Where do they fit?")
    
    # VISUALIZATION: Radar Chart or Scatter Plot
    # Let's do a Scatter Plot of Spend vs Rating (the two most defining features)
    
    # 1. Prepare Centers Data
    plot_data = cluster_means.reset_index()
    plot_data['Persona'] = plot_data['Cluster'].map(lambda x: cluster_mapping[x]['name'])
    
    # 2. Add Current User as a new point
    user_point = pd.DataFrame([{
        'Total_Amount': total_amount,
        'Customer_Rating': rating,
        'Persona': '🔴 THIS CUSTOMER',
        'Size': 15 # Make user dot bigger
    }])
    
    # Assign default size to clusters
    plot_data['Size'] = 10 
    
    # Combine
    final_plot_data = pd.concat([plot_data, user_point], ignore_index=True)
    
    fig = px.scatter(
        final_plot_data,
        x='Total_Amount',
        y='Customer_Rating',
        color='Persona',
        size='Size',
        hover_data=['Persona'],
        title="Customer Positioning (Spend vs. Satisfaction)",
        template="plotly_white",
        size_max=20
    )
    
    # Add annotation for the user
    fig.add_annotation(
        x=total_amount,
        y=rating,
        text="Current User",
        showarrow=True,
        arrowhead=1
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 5. Feature Breakdown (Optional)
# ---------------------------------------------------------
with st.expander("📊 See Detailed Feature Comparison"):
    # Compare User vs Cluster Average
    cluster_avg = cluster_means.loc[predicted_cluster]
    
    comp_df = pd.DataFrame({
        'Feature': features,
        'User Value': input_df.iloc[0].values,
        'Cluster Avg': cluster_avg.values
    })
    
    # Calculate difference
    comp_df['Diff %'] = ((comp_df['User Value'] - comp_df['Cluster Avg']) / comp_df['Cluster Avg']) * 100
    
    # FIX: Use 'subset' to apply formatting ONLY to the numeric columns
    st.dataframe(comp_df.style.format("{:.2f}", subset=['User Value', 'Cluster Avg', 'Diff %']))