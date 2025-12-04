import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page config
st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache functions
@st.cache_data
def load_data():
    df = pd.read_csv("data/rfm_clustered.csv")
    
    # Rename PCA columns if they exist
    if 'pca1' in df.columns:
        df = df.rename(columns={'pca1': 'pca1', 'pca2': 'pca2'})
    elif 'pca 1' in df.columns:
        df = df.rename(columns={'pca 1': 'pca1', 'pca 2': 'pca2'})
    
    # Add CustomerID if not exists
    if 'CustomerID' not in df.columns:
        df['CustomerID'] = df.index
    
    # 🔹 Bangun cluster_profile langsung dari df
    cluster_profile = (
        df.groupby('Segment')
          .agg(
              Recency=('Recency', 'mean'),
              Frequency=('Frequency', 'mean'),
              Monetary=('Monetary', 'mean'),
              Customer_Count=('Segment', 'size')
          )
          .round(2)
          .reset_index()
    )
    
    return df, cluster_profile


@st.cache_resource
def load_model():
    model = joblib.load("model/kmeans_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

# Helper functions
def create_rfm_radar(segment_data, segment_name):
    """Create radar chart for RFM metrics"""
    categories = ['Recency', 'Frequency', 'Monetary']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[segment_data['Recency'].mean(), 
           segment_data['Frequency'].mean(), 
           segment_data['Monetary'].mean()],
        theta=categories,
        fill='toself',
        name=segment_name
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=400,
        title=f"RFM Profile - {segment_name}"
    )
    return fig

def create_segment_comparison():
    """Create comprehensive segment comparison"""
    comparison = df.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).round(2)
    comparison.columns = ['Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Count']
    return comparison

def predict_segment(recency, frequency, monetary):
    """Predict segment for new customer"""
    input_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })
    scaled_data = scaler.transform(input_data)
    cluster = model.predict(scaled_data)[0]
    segment = df[df['Cluster'] == cluster]['Segment'].iloc[0] if 'Cluster' in df.columns else f"Cluster {cluster}"
    return segment, cluster

# Load data
try:
    df, cluster_profile = load_data()
    model, scaler = load_model()
    
    # Debug: Show column names
    #st.sidebar.info(f"Available columns: {df.columns.tolist()}")
    
    # Handle case-insensitive column names and reset index if needed
    df.columns = df.columns.str.strip()  # Remove whitespace
    
    # Check if RFM columns exist (case-insensitive)
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'recency' in col_lower:
            column_mapping[col] = 'Recency'
        elif 'frequency' in col_lower:
            column_mapping[col] = 'Frequency'
        elif 'monetary' in col_lower:
            column_mapping[col] = 'Monetary'
        elif 'segment' in col_lower:
            column_mapping[col] = 'Segment'
        elif 'customer' in col_lower and 'id' in col_lower:
            column_mapping[col] = 'CustomerID'
    
    # Rename columns if mapping found
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    # If Recency still not found, try to reset index
    if 'Recency' not in df.columns and 'Recency' in df.index.names:
        df = df.reset_index()
    
    # Verify essential columns exist
    required_cols = ['Recency', 'Frequency', 'Monetary', 'Segment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.error(f"Available columns: {df.columns.tolist()}")
        st.stop()
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    #st.image("https://via.placeholder.com/250x80/667eea/ffffff?text=RFM+Analytics", use_container_width=True)
    st.title("⚙️ Controls")
    
    # Filters
    st.subheader("📊 Filters")
    selected_segments = st.multiselect(
        "Select Segments",
        options=sorted(df['Segment'].unique()),
        default=sorted(df['Segment'].unique())
    )
    
    recency_range = st.slider(
        "Recency (days)",
        int(df['Recency'].min()),
        int(df['Recency'].max()),
        (int(df['Recency'].min()), int(df['Recency'].max()))
    )
    
    monetary_range = st.slider(
        "Monetary Value",
        float(df['Monetary'].min()),
        float(df['Monetary'].max()),
        (float(df['Monetary'].min()), float(df['Monetary'].max()))
    )
    
    st.write("---")
    st.subheader("📥 Export")
    
    if st.button("Download Filtered Data", use_container_width=True):
        filtered = df[
            (df['Segment'].isin(selected_segments)) &
            (df['Recency'].between(*recency_range)) &
            (df['Monetary'].between(*monetary_range))
        ]
        csv = filtered.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv,
            "filtered_customers.csv",
            "text/csv",
            use_container_width=True
        )

# Apply filters
filtered_df = df[
    (df['Segment'].isin(selected_segments)) &
    (df['Recency'].between(*recency_range)) &
    (df['Monetary'].between(*monetary_range))
]

# Main header
st.markdown('<p class="main-header">🎯 Customer Segmentation Analytics Platform</p>', unsafe_allow_html=True)
st.markdown("**Advanced RFM Analysis with K-Means Clustering**")
st.write("---")

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        f"{len(filtered_df):,}",
        f"{len(filtered_df) - len(df):+,}" if len(filtered_df) != len(df) else "All data"
    )

with col2:
    avg_monetary = filtered_df['Monetary'].mean()
    st.metric(
        "Avg Customer Value",
        f"${avg_monetary:,.0f}",
        f"{(avg_monetary/df['Monetary'].mean() - 1)*100:+.1f}%"
    )

with col3:
    avg_frequency = filtered_df['Frequency'].mean()
    st.metric(
        "Avg Purchase Frequency",
        f"{avg_frequency:.1f}",
        f"{(avg_frequency/df['Frequency'].mean() - 1)*100:+.1f}%"
    )

with col4:
    st.metric(
        "Active Segments",
        len(filtered_df['Segment'].unique()),
        f"of {len(df['Segment'].unique())} total"
    )

st.write("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🎯 Segment Deep Dive", 
    "📈 Advanced Analytics",
    "🔮 Predict Segment",
    "💡 Recommendations"
])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segment Distribution")
        segment_counts = filtered_df['Segment'].value_counts()
        fig_pie = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Customer Distribution by Segment",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Segment Value Contribution")
        value_by_segment = filtered_df.groupby('Segment')['Monetary'].sum().sort_values(ascending=True)
        fig_bar = px.bar(
            x=value_by_segment.values,
            y=value_by_segment.index,
            orientation='h',
            title="Total Revenue by Segment",
            labels={'x': 'Total Revenue ($)', 'y': 'Segment'},
            color=value_by_segment.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.subheader("Cluster Profile Summary")
    st.dataframe(
        cluster_profile.style.background_gradient(cmap='RdYlGn_r', subset=['Recency'])
                            .background_gradient(cmap='RdYlGn', subset=['Frequency', 'Monetary'])
                            .format({'Recency': '{:.0f}', 'Frequency': '{:.2f}', 'Monetary': '${:,.0f}'}),
        use_container_width=True
    )

with tab2:
    st.subheader("🔍 Segment Explorer")
    
    selected_segment = st.selectbox(
        "Choose a segment to analyze:",
        sorted(filtered_df['Segment'].unique())
    )
    
    segment_data = filtered_df[filtered_df['Segment'] == selected_segment]
    
    # Segment KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", f"{len(segment_data):,}")
    with col2:
        st.metric("Avg Recency", f"{segment_data['Recency'].mean():.0f} days")
    with col3:
        st.metric("Avg Frequency", f"{segment_data['Frequency'].mean():.1f}")
    with col4:
        st.metric("Avg Monetary", f"${segment_data['Monetary'].mean():,.0f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RFM Distribution
        fig_rfm = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Recency', 'Frequency', 'Monetary')
        )
        
        fig_rfm.add_trace(
            go.Histogram(x=segment_data['Recency'], name='Recency', marker_color='#667eea'),
            row=1, col=1
        )
        fig_rfm.add_trace(
            go.Histogram(x=segment_data['Frequency'], name='Frequency', marker_color='#764ba2'),
            row=1, col=2
        )
        fig_rfm.add_trace(
            go.Histogram(x=segment_data['Monetary'], name='Monetary', marker_color='#f093fb'),
            row=1, col=3
        )
        
        fig_rfm.update_layout(height=400, showlegend=False, title_text="RFM Distributions")
        st.plotly_chart(fig_rfm, use_container_width=True)
    
    with col2:
        # Radar chart
        radar_fig = create_rfm_radar(segment_data, selected_segment)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    # Customer table
    st.subheader(f"Customers in {selected_segment}")
    display_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    display_df = segment_data[display_cols].sort_values('Monetary', ascending=False)
    
    st.dataframe(
        display_df.style.format({'Monetary': '${:,.0f}', 'Frequency': '{:.1f}', 'Recency': '{:.0f}'}),
        use_container_width=True,
        height=400
    )

with tab3:
    st.subheader("📊 Advanced Segment Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D Scatter
        st.markdown("**3D RFM Visualization**")
        fig_3d = px.scatter_3d(
            filtered_df,
            x='Recency',
            y='Frequency',
            z='Monetary',
            color='Segment',
            size='Monetary',
            hover_data=['CustomerID'],
            title="3D RFM Cluster Visualization"
        )
        fig_3d.update_layout(height=500)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        # PCA Scatter
        st.markdown("**PCA Dimensionality Reduction**")
        if 'pca1' in filtered_df.columns and 'pca2' in filtered_df.columns:
            fig_pca = px.scatter(
                filtered_df,
                x='pca1',
                y='pca2',
                color='Segment',
                size='Monetary',
                hover_data=['CustomerID', 'Recency', 'Frequency', 'Monetary'],
                title="PCA Component Analysis"
            )
            fig_pca.update_layout(height=500)
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            st.info("PCA columns not found in dataset")
    
    # Segment comparison heatmap
    st.subheader("Segment Comparison Matrix")
    comparison_df = create_segment_comparison()
    
    fig_heatmap = px.imshow(
        comparison_df[['Avg Recency', 'Avg Frequency', 'Avg Monetary']].T,
        labels=dict(x="Segment", y="Metric", color="Value"),
        x=comparison_df.index,
        y=['Avg Recency', 'Avg Frequency', 'Avg Monetary'],
        aspect="auto",
        color_continuous_scale='RdYlGn'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab4:
    st.subheader("🔮 Customer Segment Predictor")
    st.markdown("Enter RFM values to predict which segment a customer belongs to:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_recency = st.number_input(
            "Recency (days)",
            min_value=0,
            max_value=int(df['Recency'].max()),
            value=int(df['Recency'].median())
        )
    
    with col2:
        pred_frequency = st.number_input(
            "Frequency (purchases)",
            min_value=0.0,
            max_value=float(df['Frequency'].max()),
            value=float(df['Frequency'].median())
        )
    
    with col3:
        pred_monetary = st.number_input(
            "Monetary ($)",
            min_value=0.0,
            max_value=float(df['Monetary'].max()),
            value=float(df['Monetary'].median())
        )
    
    if st.button("🎯 Predict Segment", use_container_width=True, type="primary"):
        predicted_segment, cluster_id = predict_segment(pred_recency, pred_frequency, pred_monetary)
        
        st.success(f"### Predicted Segment: **{predicted_segment}**")
        
        # Show segment profile
        segment_profile = cluster_profile[cluster_profile['Segment'] == predicted_segment]
        if not segment_profile.empty:
            st.info(f"**Segment Characteristics:**\n\n{segment_profile.iloc[0].to_dict()}")
        
        # Visualization
        comparison_data = pd.DataFrame({
            'Metric': ['Recency', 'Frequency', 'Monetary'],
            'Input': [pred_recency, pred_frequency, pred_monetary],
            'Segment Avg': [
                df[df['Segment'] == predicted_segment]['Recency'].mean(),
                df[df['Segment'] == predicted_segment]['Frequency'].mean(),
                df[df['Segment'] == predicted_segment]['Monetary'].mean()
            ]
        })
        
        fig_comparison = px.bar(
            comparison_data,
            x='Metric',
            y=['Input', 'Segment Avg'],
            barmode='group',
            title="Your Input vs Segment Average"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

with tab5:
    st.subheader("💡 Strategic Recommendations by Segment")
    
    recommendations = {
        "Premium Loyalists": {
            "icon": "🏆",
            "description": "Your best customers - high value, frequent purchases, recent activity",
            "strategies": [
                "Reward with VIP programs and exclusive early access",
                "Seek testimonials and referrals",
                "Test premium products and new features",
                "Implement loyalty retention programs"
            ]
        },
        "Active Mid Spenders": {
            "icon": "💎",
            "description": "High frequency buyers with good monetary value",
            "strategies": [
                "Upsell premium products and bundles",
                "Create personalized product recommendations",
                "Offer subscription or membership programs",
                "Send appreciation campaigns"
            ]
        },
        "Dormant Low Spenders": {
            "icon": "⚠️",
            "description": "Previously valuable customers showing declining activity",
            "strategies": [
                "Launch win-back campaigns with special offers",
                "Conduct surveys to understand pain points",
                "Provide personalized re-engagement incentives",
                "Offer exclusive 'we miss you' discounts"
            ]
        },
        "Lost": {
            "icon": "😔",
            "description": "Customers who haven't purchased in a long time",
            "strategies": [
                "Send re-activation campaigns with strong incentives",
                "Update them on new products/improvements",
                "Consider whether to continue marketing spend",
                "Analyze exit patterns for future prevention"
            ]
        }
    }
    
    for segment, details in recommendations.items():
        if segment in df['Segment'].unique():
            with st.expander(f"{details['icon']} {segment} - {details['description']}", expanded=True):
                st.markdown(f"**Action Strategies:**")
                for strategy in details['strategies']:
                    st.markdown(f"- {strategy}")
                
                segment_size = len(df[df['Segment'] == segment])
                segment_value = df[df['Segment'] == segment]['Monetary'].sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Segment Size", f"{segment_size:,} customers")
                with col2:
                    st.metric("Total Value", f"${segment_value:,.0f}")

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Customer Segmentation Analytics Platform</strong></p>
    <p>Powered by RFM Analysis & K-Means Clustering | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)