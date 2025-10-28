"""
Streamlit App for Neural Network Architecture Explorer
Interactive visualization and comparison of experiment results
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import numpy as np

# Page configuration
st.set_page_config(
    page_title="NN Architecture Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_all_results():
    """Load all experiment results"""
    results_dir = Path('results')
    if not results_dir.exists():
        return None, None
    
    pytorch_summary = results_dir / 'pytorch_summary.csv'
    tensorflow_summary = results_dir / 'tensorflow_summary.csv'
    
    pytorch_df = pd.read_csv(pytorch_summary) if pytorch_summary.exists() else None
    tensorflow_df = pd.read_csv(tensorflow_summary) if tensorflow_summary.exists() else None
    
    return pytorch_df, tensorflow_df

def load_experiment_detail(experiment_name):
    """Load detailed results for a specific experiment"""
    result_file = Path(f'results/{experiment_name}.json')
    if result_file.exists():
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

def main():
    st.markdown('<p class="main-header">🧠 Neural Network Architecture Explorer</p>', unsafe_allow_html=True)
    st.markdown("**Experiment with depth, width, and hyperparameters to understand their impact on model performance**")
    
    # Load data
    pytorch_df, tensorflow_df = load_all_results()
    
    if pytorch_df is None and tensorflow_df is None:
        st.error("⚠️ No experiment results found. Please run experiments first!")
        st.code("python experiment_pytorch.py\npython experiment_tensorflow.py")
        return
    
    # Combine dataframes
    all_results = []
    if pytorch_df is not None:
        all_results.append(pytorch_df)
    if tensorflow_df is not None:
        all_results.append(tensorflow_df)
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["📊 Overview", "🔍 Experiment Comparison", "📈 Depth Analysis", 
         "📏 Width Analysis", "⚙️ Hyperparameter Impact", "🏆 Best Models"]
    )
    
    if page == "📊 Overview":
        show_overview(combined_df, pytorch_df, tensorflow_df)
    elif page == "🔍 Experiment Comparison":
        show_comparison(combined_df)
    elif page == "📈 Depth Analysis":
        show_depth_analysis(combined_df)
    elif page == "📏 Width Analysis":
        show_width_analysis(combined_df)
    elif page == "⚙️ Hyperparameter Impact":
        show_hyperparameter_impact(combined_df)
    elif page == "🏆 Best Models":
        show_best_models(combined_df)

def show_overview(combined_df, pytorch_df, tensorflow_df):
    """Show overview of all experiments"""
    st.header("📊 Experiment Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Experiments", len(combined_df))
    with col2:
        best_rmse = combined_df['test_rmse'].min()
        st.metric("Best RMSE", f"{best_rmse:.4f}")
    with col3:
        best_r2 = combined_df['test_r2'].max()
        st.metric("Best R²", f"{best_r2:.4f}")
    with col4:
        avg_time = combined_df['training_time'].mean()
        st.metric("Avg Training Time", f"{avg_time:.1f}s")
    
    # Framework comparison
    st.subheader("Framework Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance comparison
        framework_perf = combined_df.groupby('framework').agg({
            'test_rmse': 'mean',
            'test_r2': 'mean',
            'training_time': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=framework_perf['framework'],
            y=framework_perf['test_rmse'],
            name='Avg RMSE',
            marker_color='indianred'
        ))
        fig.update_layout(
            title="Average RMSE by Framework",
            xaxis_title="Framework",
            yaxis_title="RMSE",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=framework_perf['framework'],
            y=framework_perf['test_r2'],
            name='Avg R²',
            marker_color='lightseagreen'
        ))
        fig.update_layout(
            title="Average R² by Framework",
            xaxis_title="Framework",
            yaxis_title="R²",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # All experiments table
    st.subheader("All Experiments")
    
    # Format the dataframe for display
    display_df = combined_df[['experiment', 'framework', 'depth', 'hidden_dims', 
                               'total_params', 'test_rmse', 'test_r2', 'training_time']].copy()
    display_df['training_time'] = display_df['training_time'].round(2)
    display_df['test_rmse'] = display_df['test_rmse'].round(4)
    display_df['test_r2'] = display_df['test_r2'].round(4)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )

def show_comparison(combined_df):
    """Compare selected experiments"""
    st.header("🔍 Experiment Comparison")
    
    # Select experiments to compare
    experiments = st.multiselect(
        "Select experiments to compare",
        combined_df['experiment'].tolist(),
        default=combined_df['experiment'].tolist()[:5]
    )
    
    if not experiments:
        st.warning("Please select at least one experiment")
        return
    
    filtered_df = combined_df[combined_df['experiment'].isin(experiments)]
    
    # Metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        for framework in filtered_df['framework'].unique():
            df_fw = filtered_df[filtered_df['framework'] == framework]
            fig.add_trace(go.Bar(
                name=framework,
                x=df_fw['experiment'],
                y=df_fw['test_rmse']
            ))
        fig.update_layout(
            title="RMSE Comparison",
            xaxis_title="Experiment",
            yaxis_title="RMSE",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        for framework in filtered_df['framework'].unique():
            df_fw = filtered_df[filtered_df['framework'] == framework]
            fig.add_trace(go.Bar(
                name=framework,
                x=df_fw['experiment'],
                y=df_fw['test_r2']
            ))
        fig.update_layout(
            title="R² Comparison",
            xaxis_title="Experiment",
            yaxis_title="R²",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Parameters vs Performance
    st.subheader("Parameters vs Performance")
    
    fig = px.scatter(
        filtered_df,
        x='total_params',
        y='test_rmse',
        color='framework',
        size='training_time',
        hover_data=['experiment', 'depth', 'hidden_dims'],
        title="Model Complexity vs Performance",
        labels={'total_params': 'Total Parameters', 'test_rmse': 'RMSE'}
    )
    st.plotly_chart(fig, use_container_width=True)

def show_depth_analysis(combined_df):
    """Analyze impact of network depth"""
    st.header("📈 Network Depth Analysis")
    
    st.markdown("""
    **Network Depth** refers to the number of hidden layers in the neural network.
    Deeper networks can learn more complex patterns but may suffer from:
    - Vanishing/exploding gradients
    - Longer training times
    - Overfitting (without proper regularization)
    """)
    
    # Group by depth
    depth_stats = combined_df.groupby('depth').agg({
        'test_rmse': ['mean', 'std', 'min'],
        'test_r2': ['mean', 'std', 'max'],
        'training_time': 'mean',
        'total_params': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=depth_stats['depth'],
            y=depth_stats['test_rmse']['mean'],
            mode='lines+markers',
            name='Mean RMSE',
            error_y=dict(
                type='data',
                array=depth_stats['test_rmse']['std'],
                visible=True
            )
        ))
        fig.update_layout(
            title="Depth vs RMSE",
            xaxis_title="Network Depth",
            yaxis_title="RMSE",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=depth_stats['depth'],
            y=depth_stats['test_r2']['mean'],
            mode='lines+markers',
            name='Mean R²',
            line=dict(color='green'),
            error_y=dict(
                type='data',
                array=depth_stats['test_r2']['std'],
                visible=True
            )
        ))
        fig.update_layout(
            title="Depth vs R²",
            xaxis_title="Network Depth",
            yaxis_title="R²",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=depth_stats['depth'],
            y=depth_stats['training_time']['mean'],
            marker_color='coral'
        ))
        fig.update_layout(
            title="Depth vs Training Time",
            xaxis_title="Network Depth",
            yaxis_title="Training Time (s)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=depth_stats['depth'],
            y=depth_stats['total_params']['mean'],
            marker_color='purple'
        ))
        fig.update_layout(
            title="Depth vs Model Parameters",
            xaxis_title="Network Depth",
            yaxis_title="Total Parameters",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.subheader("💡 Key Insights")
    best_depth = depth_stats.loc[depth_stats['test_rmse']['mean'].idxmin(), 'depth']
    st.info(f"✓ Best performing depth: **{best_depth} layers**")
    
    if len(depth_stats) > 1:
        time_increase = (depth_stats['training_time']['mean'].iloc[-1] / 
                        depth_stats['training_time']['mean'].iloc[0])
        st.info(f"✓ Training time increases by **{time_increase:.1f}x** from shallowest to deepest")

def show_width_analysis(combined_df):
    """Analyze impact of network width"""
    st.header("📏 Network Width Analysis")
    
    st.markdown("""
    **Network Width** refers to the number of neurons in each layer.
    Wider networks can capture more features but:
    - Require more parameters (memory)
    - Take longer to train
    - May overfit without regularization
    """)
    
    # Extract width from hidden_dims (average neurons per layer)
    def get_avg_width(hidden_dims_str):
        dims = eval(hidden_dims_str)
        return sum(dims) / len(dims) if dims else 0
    
    combined_df['avg_width'] = combined_df['hidden_dims'].apply(get_avg_width)
    
    # Group by average width
    width_bins = [0, 32, 100, 300, 600]
    width_labels = ['Narrow (<32)', 'Medium (32-100)', 'Wide (100-300)', 'Very Wide (>300)']
    combined_df['width_category'] = pd.cut(combined_df['avg_width'], bins=width_bins, labels=width_labels)
    
    width_stats = combined_df.groupby('width_category').agg({
        'test_rmse': 'mean',
        'test_r2': 'mean',
        'training_time': 'mean',
        'total_params': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=width_stats['width_category'],
            y=width_stats['test_rmse'],
            marker_color='indianred'
        ))
        fig.update_layout(
            title="Width Category vs RMSE",
            xaxis_title="Width Category",
            yaxis_title="RMSE",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=width_stats['width_category'],
            y=width_stats['test_r2'],
            marker_color='lightseagreen'
        ))
        fig.update_layout(
            title="Width Category vs R²",
            xaxis_title="Width Category",
            yaxis_title="R²",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter: Width vs Performance
    st.subheader("Width vs Performance (Detailed)")
    
    fig = px.scatter(
        combined_df,
        x='avg_width',
        y='test_rmse',
        color='framework',
        size='depth',
        hover_data=['experiment', 'hidden_dims'],
        title="Average Layer Width vs RMSE",
        labels={'avg_width': 'Average Neurons per Layer', 'test_rmse': 'RMSE'}
    )
    st.plotly_chart(fig, use_container_width=True)

def show_hyperparameter_impact(combined_df):
    """Analyze impact of different hyperparameters"""
    st.header("⚙️ Hyperparameter Impact")
    
    # Activation functions
    st.subheader("Activation Functions")
    
    activation_stats = combined_df.groupby('activation').agg({
        'test_rmse': 'mean',
        'test_r2': 'mean',
        'training_time': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=activation_stats['activation'],
            y=activation_stats['test_rmse'],
            marker_color='skyblue'
        ))
        fig.update_layout(
            title="Activation Function vs RMSE",
            xaxis_title="Activation Function",
            yaxis_title="RMSE",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=activation_stats['activation'],
            y=activation_stats['test_r2'],
            marker_color='lightgreen'
        ))
        fig.update_layout(
            title="Activation Function vs R²",
            xaxis_title="Activation Function",
            yaxis_title="R²",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Learning rate
    st.subheader("Learning Rate")
    
    lr_stats = combined_df.groupby('learning_rate').agg({
        'test_rmse': 'mean',
        'test_r2': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lr_stats['learning_rate'],
            y=lr_stats['test_rmse'],
            mode='lines+markers',
            line=dict(color='red')
        ))
        fig.update_layout(
            title="Learning Rate vs RMSE",
            xaxis_title="Learning Rate",
            yaxis_title="RMSE",
            xaxis_type="log",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lr_stats['learning_rate'],
            y=lr_stats['test_r2'],
            mode='lines+markers',
            line=dict(color='green')
        ))
        fig.update_layout(
            title="Learning Rate vs R²",
            xaxis_title="Learning Rate",
            yaxis_title="R²",
            xaxis_type="log",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Regularization
    st.subheader("Regularization Techniques")
    
    reg_experiments = combined_df[combined_df['experiment'].str.contains('dropout|l2', case=False)]
    
    if len(reg_experiments) > 0:
        baseline_experiments = combined_df[~combined_df['experiment'].str.contains('dropout|l2', case=False)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=baseline_experiments['test_rmse'],
                name='Without Regularization',
                marker_color='lightcoral'
            ))
            fig.add_trace(go.Box(
                y=reg_experiments['test_rmse'],
                name='With Regularization',
                marker_color='lightgreen'
            ))
            fig.update_layout(
                title="Regularization Impact on RMSE",
                yaxis_title="RMSE",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=baseline_experiments['test_r2'],
                name='Without Regularization',
                marker_color='lightcoral'
            ))
            fig.add_trace(go.Box(
                y=reg_experiments['test_r2'],
                name='With Regularization',
                marker_color='lightgreen'
            ))
            fig.update_layout(
                title="Regularization Impact on R²",
                yaxis_title="R²",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

def show_best_models(combined_df):
    """Show best performing models"""
    st.header("🏆 Best Models")
    
    # Best by different metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🥇 Best RMSE")
        best_rmse = combined_df.loc[combined_df['test_rmse'].idxmin()]
        st.metric("Experiment", best_rmse['experiment'])
        st.metric("RMSE", f"{best_rmse['test_rmse']:.4f}")
        st.metric("Framework", best_rmse['framework'])
        st.metric("Architecture", best_rmse['hidden_dims'])
    
    with col2:
        st.subheader("🥇 Best R²")
        best_r2 = combined_df.loc[combined_df['test_r2'].idxmax()]
        st.metric("Experiment", best_r2['experiment'])
        st.metric("R²", f"{best_r2['test_r2']:.4f}")
        st.metric("Framework", best_r2['framework'])
        st.metric("Architecture", best_r2['hidden_dims'])
    
    with col3:
        st.subheader("⚡ Fastest Training")
        fastest = combined_df.loc[combined_df['training_time'].idxmin()]
        st.metric("Experiment", fastest['experiment'])
        st.metric("Time", f"{fastest['training_time']:.2f}s")
        st.metric("Framework", fastest['framework'])
        st.metric("RMSE", f"{fastest['test_rmse']:.4f}")
    
    # Pareto frontier: Performance vs Complexity
    st.subheader("🎯 Performance vs Complexity (Pareto Frontier)")
    
    st.markdown("""
    The **Pareto frontier** shows models that are not strictly dominated by any other model.
    These represent the best trade-offs between performance and model complexity.
    """)
    
    # Calculate Pareto frontier
    def is_pareto_efficient(costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient
    
    # Use RMSE (lower is better) and parameters (lower is better)
    costs = np.column_stack((combined_df['test_rmse'].values, combined_df['total_params'].values))
    pareto_mask = is_pareto_efficient(costs)
    
    combined_df['is_pareto'] = pareto_mask
    
    fig = px.scatter(
        combined_df,
        x='total_params',
        y='test_rmse',
        color='is_pareto',
        symbol='framework',
        hover_data=['experiment', 'depth', 'hidden_dims'],
        title="Performance vs Model Complexity",
        labels={'total_params': 'Total Parameters', 'test_rmse': 'RMSE'},
        color_discrete_map={True: 'gold', False: 'lightgray'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 models
    st.subheader("📋 Top 5 Models by RMSE")
    
    top_5 = combined_df.nsmallest(5, 'test_rmse')[
        ['experiment', 'framework', 'depth', 'hidden_dims', 'total_params', 
         'test_rmse', 'test_r2', 'training_time']
    ].copy()
    
    top_5['training_time'] = top_5['training_time'].round(2)
    top_5['test_rmse'] = top_5['test_rmse'].round(4)
    top_5['test_r2'] = top_5['test_r2'].round(4)
    
    st.dataframe(top_5, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    best_overall = combined_df.loc[combined_df['test_rmse'].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **For Best Performance:**
        - Use **{best_overall['experiment']}** configuration
        - Framework: **{best_overall['framework']}**
        - Architecture: **{best_overall['hidden_dims']}**
        - Expected RMSE: **{best_overall['test_rmse']:.4f}**
        """)
    
    with col2:
        fastest_good = combined_df[combined_df['test_rmse'] < combined_df['test_rmse'].quantile(0.3)].loc[
            combined_df[combined_df['test_rmse'] < combined_df['test_rmse'].quantile(0.3)]['training_time'].idxmin()
        ]
        st.success(f"""
        **For Speed with Good Performance:**
        - Use **{fastest_good['experiment']}** configuration
        - Framework: **{fastest_good['framework']}**
        - Training Time: **{fastest_good['training_time']:.2f}s**
        - Expected RMSE: **{fastest_good['test_rmse']:.4f}**
        """)

if __name__ == "__main__":
    main()