import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Composer DB Evaluation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.kpi-card {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem;
    text-align: center;
}
.algo-name {
    font-weight: bold;
    color: #2c3e50;
}
.stTabs > div > div > div > div {
    gap: 2rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_from_file(uploaded_file):
    """Load trading algorithm data from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file, engine='python')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# File upload section
st.header("📁 Data Upload")
uploaded_file = st.file_uploader(
    "Upload your trading algorithm CSV file",
    type=['csv'],
    help="Upload a CSV file containing trading algorithm performance data with columns like oos_sharpe_ratio, oos_annualized_rate_of_return, etc."
)

# Load data only if file is uploaded
if uploaded_file is not None:
    df = load_data_from_file(uploaded_file)
    
    if df.empty:
        st.error("Failed to load data from the uploaded file. Please check the file format.")
        st.stop()
    
    st.success(f"✅ Successfully loaded {len(df)} algorithms with {len(df.columns)} metrics")
else:
    st.info("👆 Please upload a CSV file to begin analyzing trading algorithms.")
    st.markdown("""
    **Expected CSV format:**
    - `symphony_sid`: Unique algorithm identifier
    - `name`: Algorithm name
    - `oos_sharpe_ratio`: Out-of-sample Sharpe ratio
    - `oos_annualized_rate_of_return`: Out-of-sample annual return
    - `oos_max_drawdown`: Out-of-sample maximum drawdown
    - `train_sharpe_ratio`: Training period Sharpe ratio
    - `asset_classes`: Asset classification
    - And other performance metrics...
    """)
    st.stop()

# Header
st.title("📈 Composer DB Evaluation Dashboard")
st.markdown("**Comprehensive analytics for quantitative trading algorithm performance**")

# Sidebar - Filters
st.sidebar.header("🔍 Filters")

# Asset class filter
if 'asset_classes' in df.columns:
    asset_classes = df['asset_classes'].dropna().unique()
    selected_assets = st.sidebar.multiselect("Asset Classes", asset_classes, default=asset_classes)
    df_filtered = df[df['asset_classes'].isin(selected_assets)] if selected_assets else df
else:
    df_filtered = df

# Rebalance frequency filter
if 'rebalance_frequency' in df.columns:
    rebal_freq = df_filtered['rebalance_frequency'].dropna().unique()
    selected_rebal = st.sidebar.multiselect("Rebalance Frequency", rebal_freq, default=rebal_freq)
    df_filtered = df_filtered[df_filtered['rebalance_frequency'].isin(selected_rebal)] if selected_rebal else df_filtered

# Performance filters
st.sidebar.subheader("Performance Thresholds")
min_sharpe = st.sidebar.slider("Min Sharpe Ratio", 
                               float(df_filtered['oos_sharpe_ratio'].min()), 
                               float(df_filtered['oos_sharpe_ratio'].max()), 
                               float(df_filtered['oos_sharpe_ratio'].min()))

max_drawdown = st.sidebar.slider("Max Drawdown Threshold", 
                                float(df_filtered['oos_max_drawdown'].min()), 
                                float(df_filtered['oos_max_drawdown'].max()),
                                float(df_filtered['oos_max_drawdown'].max()))

df_filtered = df_filtered[
    (df_filtered['oos_sharpe_ratio'] >= min_sharpe) & 
    (df_filtered['oos_max_drawdown'] <= max_drawdown)
]

# KPI Cards
st.header("📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    best_sharpe = df_filtered['oos_sharpe_ratio'].max()
    best_sharpe_algo = df_filtered.loc[df_filtered['oos_sharpe_ratio'].idxmax(), 'name']
    st.metric(
        "🏆 Best Sharpe Ratio", 
        f"{best_sharpe:.3f}",
        help=f"Algorithm: {best_sharpe_algo}"
    )

with col2:
    best_return = df_filtered['oos_annualized_rate_of_return'].max()
    best_return_algo = df_filtered.loc[df_filtered['oos_annualized_rate_of_return'].idxmax(), 'name']
    st.metric(
        "💰 Best Annual Return", 
        f"{best_return:.1%}",
        help=f"Algorithm: {best_return_algo}"
    )

with col3:
    lowest_dd = df_filtered['oos_max_drawdown'].min()
    lowest_dd_algo = df_filtered.loc[df_filtered['oos_max_drawdown'].idxmin(), 'name']
    st.metric(
        "🛡️ Lowest Drawdown", 
        f"{lowest_dd:.1%}",
        help=f"Algorithm: {lowest_dd_algo}"
    )

with col4:
    st.metric("🔢 Total Algorithms", len(df_filtered))

# Main Dashboard Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Leaderboard", 
    "📊 Risk vs Return", 
    "⚖️ Train vs OOS", 
    "📈 Performance Horizons",
    "🎯 Asset Analysis",
    "🔍 Algorithm Details"
])

with tab1:
    st.subheader("🏆 Algorithm Leaderboard")
    
    # Create leaderboard dataframe
    leaderboard_cols = [
        'symphony_sid', 'name', 'asset_classes', 'rebalance_frequency',
        'oos_sharpe_ratio', 'oos_annualized_rate_of_return', 'oos_max_drawdown', 
        'oos_cumulative_return'
    ]
    
    leaderboard_df = df_filtered[leaderboard_cols].copy()
    
    # Create full URLs for Symphony IDs
    leaderboard_df['composer_url'] = leaderboard_df['symphony_sid'].apply(
        lambda x: f"https://app.composer.trade/symphony/{x}/details"
    )
    
    # Format numeric columns
    leaderboard_df['oos_annualized_rate_of_return'] = leaderboard_df['oos_annualized_rate_of_return'].apply(lambda x: f"{x:.1%}")
    leaderboard_df['oos_max_drawdown'] = leaderboard_df['oos_max_drawdown'].apply(lambda x: f"{x:.1%}")
    leaderboard_df['oos_sharpe_ratio'] = leaderboard_df['oos_sharpe_ratio'].round(3)
    leaderboard_df['oos_cumulative_return'] = leaderboard_df['oos_cumulative_return'].round(2)
    
    # Select and rename columns for display
    leaderboard_display = leaderboard_df[[
        'composer_url', 'name', 'asset_classes', 'rebalance_frequency',
        'oos_sharpe_ratio', 'oos_annualized_rate_of_return', 'oos_max_drawdown', 
        'oos_cumulative_return'
    ]].copy()
    
    leaderboard_display.columns = [
        'Symphony ID', 'Algorithm Name', 'Asset Class', 'Rebalance Freq',
        'Sharpe Ratio', 'Annual Return', 'Max Drawdown', 'Cumulative Return'
    ]
    
    st.dataframe(
        leaderboard_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symphony ID": st.column_config.LinkColumn(
                "Symphony ID",
                help="Click Symphony ID to view algorithm in Composer app",
                display_text=r"https://app\.composer\.trade/symphony/(.+)/details",
                width="medium"
            ),
            "Algorithm Name": st.column_config.TextColumn(width="large"),
            "Sharpe Ratio": st.column_config.NumberColumn(format="%.3f"),
            "Annual Return": st.column_config.TextColumn(width="small"),
            "Max Drawdown": st.column_config.TextColumn(width="small"),
        }
    )

with tab2:
    st.subheader("📊 Risk vs Return Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk-Return Scatter Plot
        # Convert negative drawdown to positive values for bubble size
        df_filtered_plot = df_filtered.copy()
        df_filtered_plot['abs_max_drawdown'] = abs(df_filtered_plot['oos_max_drawdown'])
        
        fig = px.scatter(
            df_filtered_plot,
            x='oos_annualized_rate_of_return',
            y='oos_sharpe_ratio',
            size='abs_max_drawdown',
            color='asset_classes' if 'asset_classes' in df_filtered.columns else None,
            hover_data=['symphony_sid', 'name', 'oos_cumulative_return', 'oos_max_drawdown'],
            title='Risk vs Return (Bubble size = Max Drawdown)',
            labels={
                'oos_annualized_rate_of_return': 'Annual Return',
                'oos_sharpe_ratio': 'Sharpe Ratio',
                'abs_max_drawdown': 'Max Drawdown (absolute)'
            }
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sharpe Ratio Distribution
        fig = px.histogram(
            df_filtered,
            x='oos_sharpe_ratio',
            nbins=15,
            title='Sharpe Ratio Distribution',
            labels={'oos_sharpe_ratio': 'Sharpe Ratio'}
        )
        fig.add_vline(
            x=df_filtered['oos_sharpe_ratio'].mean(),
            line_dash="dash",
            annotation_text="Mean"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Efficient Frontier Style View
    st.subheader("Efficient Frontier Visualization")
    
    # Ensure positive values for bubble size
    df_filtered_ef = df_filtered.copy()
    df_filtered_ef['abs_cumulative_return'] = abs(df_filtered_ef['oos_cumulative_return']) + 0.1  # Add small constant to avoid zero
    
    fig = px.scatter(
        df_filtered_ef,
        x='oos_standard_deviation',
        y='oos_annualized_rate_of_return',
        color='oos_sharpe_ratio',
        size='abs_cumulative_return',
        hover_data=['name', 'symphony_sid', 'oos_cumulative_return'],
        title='Risk (Volatility) vs Expected Return',
        labels={
            'oos_standard_deviation': 'Volatility (Standard Deviation)',
            'oos_annualized_rate_of_return': 'Expected Return',
            'abs_cumulative_return': 'Cumulative Return (absolute)'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("⚖️ Train vs Out-of-Sample Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sharpe Ratio Comparison
        comparison_df = df_filtered[['name', 'train_sharpe_ratio', 'oos_sharpe_ratio']].copy()
        comparison_df = comparison_df.melt(
            id_vars=['name'], 
            value_vars=['train_sharpe_ratio', 'oos_sharpe_ratio'],
            var_name='Period',
            value_name='Sharpe Ratio'
        )
        comparison_df['Period'] = comparison_df['Period'].map({
            'train_sharpe_ratio': 'Training',
            'oos_sharpe_ratio': 'Out-of-Sample'
        })
        
        fig = px.bar(
            comparison_df,
            x='name',
            y='Sharpe Ratio',
            color='Period',
            barmode='group',
            title='Sharpe Ratio: Train vs OOS',
            height=400
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Return Comparison
        return_comparison_df = df_filtered[['name', 'train_annualized_rate_of_return', 'oos_annualized_rate_of_return']].copy()
        return_comparison_df = return_comparison_df.melt(
            id_vars=['name'],
            value_vars=['train_annualized_rate_of_return', 'oos_annualized_rate_of_return'],
            var_name='Period',
            value_name='Annual Return'
        )
        return_comparison_df['Period'] = return_comparison_df['Period'].map({
            'train_annualized_rate_of_return': 'Training',
            'oos_annualized_rate_of_return': 'Out-of-Sample'
        })
        
        fig = px.bar(
            return_comparison_df,
            x='name',
            y='Annual Return',
            color='Period',
            barmode='group',
            title='Annual Return: Train vs OOS',
            height=400
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Overfitting Detection
    st.subheader("🚨 Overfitting Detection")
    df_filtered['sharpe_diff'] = df_filtered['train_sharpe_ratio'] - df_filtered['oos_sharpe_ratio']
    df_filtered['return_diff'] = df_filtered['train_annualized_rate_of_return'] - df_filtered['oos_annualized_rate_of_return']
    
    overfitting_df = df_filtered[['name', 'sharpe_diff', 'return_diff']].copy()
    overfitting_df = overfitting_df.sort_values('sharpe_diff', ascending=False)
    
    fig = px.bar(
        overfitting_df,
        x='name',
        y='sharpe_diff',
        title='Potential Overfitting (Train Sharpe - OOS Sharpe)',
        labels={'sharpe_diff': 'Sharpe Difference (Train - OOS)'},
        color='sharpe_diff',
        color_continuous_scale='RdYlBu_r'
    )
    fig.update_layout(xaxis_tickangle=45)
    fig.add_hline(y=0, line_dash="dash", annotation_text="No Difference")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("📈 Multi-Horizon Performance Analysis")
    
    # Trailing returns analysis
    horizon_cols = [
        'oos_trailing_one_month_return',
        'oos_trailing_three_month_return', 
        'oos_trailing_one_year_return'
    ]
    
    available_horizons = [col for col in horizon_cols if col in df_filtered.columns]
    
    if available_horizons:
        horizon_df = df_filtered[['name'] + available_horizons].copy()
        horizon_df = horizon_df.melt(
            id_vars=['name'],
            value_vars=available_horizons,
            var_name='Horizon',
            value_name='Return'
        )
        
        # Clean up horizon names
        horizon_mapping = {
            'oos_trailing_one_month_return': '1 Month',
            'oos_trailing_three_month_return': '3 Month',
            'oos_trailing_one_year_return': '1 Year'
        }
        horizon_df['Horizon'] = horizon_df['Horizon'].map(horizon_mapping)
        
        fig = px.bar(
            horizon_df,
            x='name',
            y='Return',
            color='Horizon',
            barmode='group',
            title='Trailing Returns by Horizon',
            height=500
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Asset-specific performance
    st.subheader("Asset-Specific Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        # SPY Performance
        spy_cols = ['name', 'oos_spy_sharpe_ratio', 'oos_spy_annualized_rate_of_return', 'oos_spy_max_drawdown']
        spy_available = [col for col in spy_cols if col in df_filtered.columns]
        
        if len(spy_available) > 1:
            fig = px.bar(
                df_filtered,
                x='name',
                y='oos_spy_sharpe_ratio',
                title='SPY-Correlated Performance (Sharpe)',
                height=400
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # BTC Performance
        btc_cols = ['name', 'oos_btcusd_sharpe_ratio', 'oos_btcusd_annualized_rate_of_return']
        btc_available = [col for col in btc_cols if col in df_filtered.columns]
        
        if len(btc_available) > 1:
            fig = px.bar(
                df_filtered,
                x='name',
                y='oos_btcusd_sharpe_ratio',
                title='BTCUSD-Correlated Performance (Sharpe)',
                height=400
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("🎯 Asset Class Analysis")
    
    if 'asset_classes' in df_filtered.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Asset class distribution
            asset_dist = df_filtered['asset_classes'].value_counts()
            fig = px.pie(
                values=asset_dist.values,
                names=asset_dist.index,
                title='Algorithm Distribution by Asset Class'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance by asset class
            asset_performance = df_filtered.groupby('asset_classes').agg({
                'oos_sharpe_ratio': 'mean',
                'oos_annualized_rate_of_return': 'mean',
                'oos_max_drawdown': 'mean'
            }).reset_index()
            
            fig = px.bar(
                asset_performance,
                x='asset_classes',
                y='oos_sharpe_ratio',
                title='Average Sharpe Ratio by Asset Class'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk Metrics Heatmap
    st.subheader("Risk Metrics Correlation")
    risk_metrics = ['oos_sharpe_ratio', 'oos_calmar_ratio', 'oos_max_drawdown', 'oos_standard_deviation']
    available_risk_metrics = [col for col in risk_metrics if col in df_filtered.columns]
    
    if len(available_risk_metrics) > 2:
        corr_matrix = df_filtered[available_risk_metrics].corr()
        
        fig = px.imshow(
            corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            color_continuous_scale='RdBu',
            title='Risk Metrics Correlation Matrix',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("🔍 Algorithm Details")
    
    # Algorithm selector for drill-down (moved from sidebar)
    selected_algo = st.selectbox(
        "Select Algorithm for Analysis",
        options=df_filtered['symphony_sid'].tolist(),
        format_func=lambda x: f"{df_filtered[df_filtered['symphony_sid']==x]['name'].iloc[0][:50]}..." if not pd.isna(df_filtered[df_filtered['symphony_sid']==x]['name'].iloc[0]) else x,
        help="Choose an algorithm to view detailed performance metrics and analysis"
    )
    
    # Get selected algorithm data
    algo_data = df[df['symphony_sid'] == selected_algo].iloc[0]
    
    # Algorithm Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📋 Basic Info")
        st.write(f"**Name:** {algo_data['name']}")
        
        # Create clickable Symphony ID link
        symphony_link = f"https://app.composer.trade/symphony/{algo_data['symphony_sid']}/details"
        st.markdown(f"**Symphony ID:** [{algo_data['symphony_sid']}]({symphony_link})")
        st.caption("↗️ Click Symphony ID to view in Composer app")
        if 'asset_classes' in algo_data:
            st.write(f"**Asset Class:** {algo_data['asset_classes']}")
        if 'rebalance_frequency' in algo_data:
            st.write(f"**Rebalance Frequency:** {algo_data['rebalance_frequency']}")
    
    with col2:
        st.markdown("### 📊 Key Metrics")
        st.metric("OOS Sharpe Ratio", f"{algo_data['oos_sharpe_ratio']:.3f}")
        st.metric("OOS Annual Return", f"{algo_data['oos_annualized_rate_of_return']:.1%}")
        st.metric("OOS Max Drawdown", f"{algo_data['oos_max_drawdown']:.1%}")
    
    with col3:
        st.markdown("### 🎯 Risk Metrics")
        st.metric("OOS Standard Deviation", f"{algo_data['oos_standard_deviation']:.3f}")
        if 'oos_calmar_ratio' in algo_data:
            st.metric("OOS Calmar Ratio", f"{algo_data['oos_calmar_ratio']:.3f}")
        st.metric("Cumulative Return", f"{algo_data['oos_cumulative_return']:.2f}")
    
    # Train vs OOS Detailed Comparison
    st.markdown("### ⚖️ Train vs Out-of-Sample Comparison")
    
    comparison_metrics = [
        ('Sharpe Ratio', 'train_sharpe_ratio', 'oos_sharpe_ratio'),
        ('Annual Return', 'train_annualized_rate_of_return', 'oos_annualized_rate_of_return'),
        ('Max Drawdown', 'train_max_drawdown', 'oos_max_drawdown'),
        ('Standard Deviation', 'train_standard_deviation', 'oos_standard_deviation')
    ]
    
    comparison_data = []
    for metric_name, train_col, oos_col in comparison_metrics:
        if train_col in algo_data and oos_col in algo_data:
            comparison_data.append({
                'Metric': metric_name,
                'Training': algo_data[train_col],
                'Out-of-Sample': algo_data[oos_col],
                'Difference': algo_data[train_col] - algo_data[oos_col]
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    # Node Composition (if available)
    st.markdown("### 🔗 Algorithm Composition")
    node_cols = [col for col in algo_data.index if col.startswith('num_node_')]
    
    if node_cols:
        node_data = []
        for col in node_cols:
            if algo_data[col] > 0:
                node_type = col.replace('num_node_', '').replace('_', ' ').title()
                node_data.append({'Node Type': node_type, 'Count': algo_data[col]})
        
        if node_data:
            node_df = pd.DataFrame(node_data)
            fig = px.pie(
                node_df,
                values='Count',
                names='Node Type',
                title='Algorithm Node Composition'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Algorithm Description
    if 'description' in algo_data and pd.notna(algo_data['description']):
        st.markdown("### 📝 Description")
        st.text_area("Algorithm Description", algo_data['description'], height=100)

# Footer
st.markdown("---")
st.markdown("📊 **Composer DB Evaluation Dashboard** | Advanced analytics for quantitative strategy assessment")