# Merging.py - add a show() function for Streamlit
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def show():
    st.title("Merging & Exploration")
    
    # Load merged data
    merged_df = pd.read_csv("merged_infrastructure_efficiency.csv")
    
    with st.expander("Merge Summary Statistics", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Countries", len(merged_df))
        with col2:
            elite_count = (merged_df['supercomputing_category'] == 'Elite Supercomputing Nation').sum()
            st.metric("Elite Supercomputing Nations", elite_count)
        with col3:
            infra_only = (merged_df['supercomputing_category'] == 'Infrastructure Only').sum()
            st.metric("Infrastructure Only", infra_only)
        
        st.markdown("---")
        
        # Show which countries are elite
        st.subheader("Elite Supercomputing Nations")
        elite_nations = merged_df[merged_df['supercomputing_category'] == 'Elite Supercomputing Nation'][
            ['country', 'num_supercomputers', 'avg_energy_efficiency', 
             'total_data_centers', 'power_capacity_MW_total', 
             'average_renewable_energy_usage_percent']
        ].sort_values('num_supercomputers', ascending=False)
        
        st.dataframe(elite_nations, use_container_width=True, hide_index=True)


        with st.expander("Elite vs. Infrastructure-Only Comparison", expanded=True):
            st.subheader("Comparing Infrastructure Characteristics")
            
            # Calculate summary stats for both groups
            comparison_vars = [
                'total_data_centers',
                'power_capacity_MW_total',
                'average_renewable_energy_usage_percent',
                'internet_penetration_percent',
                'growth_rate_of_data_centers_percent_per_year'
            ]
            
            comparison_df = merged_df.groupby('supercomputing_category')[comparison_vars].mean()
            
            # Display as table
            st.dataframe(comparison_df.T, use_container_width=True)
            
            # Visualize differences
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, var in enumerate(comparison_vars):
                ax = axes[idx]
                
                elite_data = merged_df[merged_df['supercomputing_category'] == 'Elite Supercomputing Nation'][var].dropna()
                infra_data = merged_df[merged_df['supercomputing_category'] == 'Infrastructure Only'][var].dropna()
                
                ax.hist([elite_data, infra_data], 
                    label=['Elite Nations', 'Infrastructure Only'],
                    alpha=0.7, 
                    bins=20,
                    color=['#2D5016', '#287AB8'])
                
                ax.set_title(var.replace('_', ' ').title(), fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with st.expander("Correlation Analysis: Elite Nations", expanded=True):
            st.subheader("How Infrastructure Relates to Efficiency (Elite Nations Only)")
            
            # Filter to elite nations only
            elite_df = merged_df[merged_df['supercomputing_category'] == 'Elite Supercomputing Nation'].copy()
            
            # Select relevant columns for correlation
            corr_vars = [
                'avg_energy_efficiency',
                'num_supercomputers',
                'total_data_centers',
                'power_capacity_MW_total',
                'average_renewable_energy_usage_percent',
                'internet_penetration_percent',
                'growth_rate_of_data_centers_percent_per_year'
            ]
            
            corr_matrix = elite_df[corr_vars].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, ax=ax, linewidths=0.5)
            ax.set_title("Correlation Matrix: Elite Supercomputing Nations", 
                        fontweight='bold', fontsize=14)
            st.pyplot(fig)
            plt.close()
            
            st.markdown("""
            **Key Questions to Explore:**
            - Does renewable energy usage correlate with energy efficiency?
            - Do countries with more data centers have more supercomputers?
            - Is there a relationship between power capacity and efficiency?
            """)


            with st.expander("Data Availability Analysis", expanded=True):
                st.subheader("Which Countries Have Complete Data?")
                
                # Check completeness
                key_vars = ['total_data_centers', 'power_capacity_MW_total', 
                        'avg_energy_efficiency', 'num_supercomputers']
                
                completeness = merged_df[key_vars].notna().sum()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                completeness.plot(kind='bar', ax=ax, color='#2D5016')
                ax.set_ylabel('Number of Countries with Data')
                ax.set_title('Data Completeness Across Variables', fontweight='bold')
                ax.axhline(y=len(merged_df), color='red', linestyle='--', 
                        label=f'Total Countries ({len(merged_df)})')
                ax.legend()
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
                plt.close()