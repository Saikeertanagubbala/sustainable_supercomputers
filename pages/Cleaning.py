import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("data_centers_original.csv")
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_columns]

@st.cache_data
def perform_knn_imputation(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]
    
    df_without_missing = df_numeric.dropna()
    
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_without_missing), columns=df_without_missing.columns)
    
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(df_scaled)
    
    def impute_and_inverse_transform(data):
        scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
        imputed_scaled = imputer.transform(scaled_data)
        return pd.DataFrame(scaler.inverse_transform(imputed_scaled), columns=data.columns, index=data.index)
    
    df_imputed = impute_and_inverse_transform(df_numeric)
    df_combined_knn = df_numeric.fillna(df_imputed)
    
    return df_combined_knn, df_numeric

@st.cache_data
def simple_imputer(df):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_columns]
    imputer.fit(df_numeric)
    imputed_array = imputer.transform(df_numeric)

    imputed_data = pd.DataFrame(imputed_array, columns=df_numeric.columns, index=df_numeric.index)
    df_combined = df_numeric.fillna(imputed_data)
    
    return df_combined, df_numeric

@st.cache_data
def plot_transformations(df, variable_name, figsize=(12, 10)):
    df_temp = df[[variable_name]].copy()
    
    df_temp['log_transform'] = np.log1p(df_temp[variable_name])
    df_temp['sqrt_transform'] = np.sqrt(df_temp[variable_name])
    df_temp['cbrt_transform'] = np.cbrt(df_temp[variable_name])
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    #Original
    sns.histplot(df_temp[variable_name], kde=True, bins=30, color='#2D5016', ax=axes[0])
    axes[0].set_title(f'Original Distribution (Skew: {df_temp[variable_name].skew():.3f})', fontweight='bold')
    axes[0].set_xlabel(variable_name)
    
    #Log Plot
    sns.histplot(df_temp['log_transform'], kde=True, bins=30, color='#5F9EA0', ax=axes[1])
    axes[1].set_title(f'Log Transformed (Skew: {df_temp["log_transform"].skew():.3f})', fontweight='bold')
    axes[1].set_xlabel(f'Log of {variable_name}')
    
    # Square Root plot
    sns.histplot(df_temp['sqrt_transform'], kde=True, bins=30, color='#4682B4', ax=axes[2])
    axes[2].set_title(f'Square Root Transformed (Skew: {df_temp["sqrt_transform"].skew():.3f})', fontweight='bold')
    axes[2].set_xlabel(f'Square Root of {variable_name}')
    
    # Cube Root Transformed
    sns.histplot(df_temp['cbrt_transform'], kde=True, bins=30, color='#6B8E23', ax=axes[3])
    axes[3].set_title(f'Cube Root Transformed (Skew: {df_temp["cbrt_transform"].skew():.3f})', 
                      fontweight='bold')
    axes[3].set_xlabel(f'Cube Root of {variable_name}')
    
    fig.suptitle(f'Different Transformations on {variable_name}', fontsize=16, fontweight='bold', y=0.99)
    
    plt.tight_layout()
    return fig

@st.cache_resource
def visualizing_missingness(df):
    columns_to_plot = df.columns.to_list()

    nan_mask = df[columns_to_plot].isna()
    nan_array = nan_mask.astype(int).to_numpy()

    plt.figure(figsize=(12, 6))
    im = plt.imshow(nan_array.T, interpolation='nearest', aspect='auto', cmap='cividis')
    plt.xlabel('Range Index')
    plt.ylabel('Features')
    plt.title('Visualizing Missing Values in Dataset')

    plt.yticks(range(len(columns_to_plot)), columns_to_plot)

    num_rows = nan_array.shape[0]
    plt.xticks(np.linspace(0, num_rows-1, min(10, num_rows)).astype(int)) 
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    return plt


def show():
    st.title("Cleaning Data")
    with st.expander("Before Cleaning", expanded=True):
        st.markdown("""
            The original datacenters dataset had several issues mainly that the variables were all stored as strings, missing values were represented inconsistently, and there were several irrelevant columns.
            \n\n I had started by dropping unecessary columns, standardizing units to what they were supposed to represent, and getting rid of extra characters.
            \n\n After that, I converted the relevant columns to numeric types.
            \n\n Initial data for this dataset then looked like this:
        """)
        st.dataframe(df.head(), width='stretch')

        st.markdown("""
            As for the Green500 dataset, due to having many features, the most that was done was dropping irrelevant columns. Most variables were already in correct types.
            \n\n Imputation did need to be done for a couple variables, which you can see moving forward. 
            \n\n Initial data for this dataset then looked like this:
            """)
        green500_df = pd.read_csv("green500_combined.csv")
        st.dataframe(green500_df.head(), width='stretch')


        
        
    with st.expander("Imputation", expanded=True):
        st.write("### Visualizing Missingness")
        st.write("Prior to imputation, I visualized missing values using a heatmap to detect any patterns.")
        
        @st.fragment
        def missingness_plot_fragment():
            my_two_datasets = {
                "Data Centers Dataset": pd.read_csv("data_centers_original.csv"),
                "Green500 Dataset": pd.read_csv("green500_combined.csv")}
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                selected_dataset = st.selectbox("Choose between datasets to view their missingness:", options=list(my_two_datasets.keys()))

            selected_df = my_two_datasets[selected_dataset]
            fig = visualizing_missingness(selected_df)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(fig, use_container_width=False)
                plt.close()
            with col2:
                missing_data = pd.DataFrame({
                    'Variable': selected_df.columns,
                    'Missing %': (selected_df.isnull().sum() / len(selected_df) * 100).values
                })
                st.dataframe(missing_data, width='stretch', hide_index=True)
        
        missingness_plot_fragment()

    

#----------------------------------------------------------------------------------------------------------------------------------#
        st.write("### Imputation Strategies (KNN vs Simple Imputer)")
        st.write("These two plots compare KNN Imputer vs Simple Imputer (median strategy) on the datacenters dataset.")
        col1, col2 = st.columns([1.5, 1.5])
        with col1:
            df_combined_knn, df_numeric = perform_knn_imputation(df)

            columns_to_plot = df_combined_knn.columns[2:]
            fig, axs = plt.subplots(3, 2, figsize=(10, 10))

            for i, ax in enumerate(axs.flatten()):
                col_name = columns_to_plot[i]
                sns.histplot(df_combined_knn[col_name], ax=ax, kde=True, color='green', bins=30, alpha=0.5, label=f'Original + Imputed = {len(df_combined_knn[col_name])}')
                sns.histplot(df_numeric[col_name].dropna(), ax=ax, kde = True, color='blue', bins = 30, alpha=0.5, label=f'Original (non-missing) = {len(df_numeric[col_name].dropna())}')
                ax.set_title(f'Dist. of {col_name} \n (KNNImputer)', fontsize=10, fontweight = 'bold')
                ax.legend()
                ax.grid(alpha = 0.5)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig, width='content')  
            plt.close() 
        
        with col2:
            df_combined, df_numeric = simple_imputer(df)

            columns_to_plot = df_combined.columns[2:]
            fig, axs = plt.subplots(3, 2, figsize=(10, 10))

            for i, ax in enumerate(axs.flatten()):
                col_name = columns_to_plot[i]
                sns.histplot(df_combined[col_name], ax=ax, kde=True, color='green', bins=30, alpha=0.5, label=f'Original + Imputed = {len(df_combined[col_name])}')
                sns.histplot(df_numeric[col_name].dropna(), ax=ax, kde = True, color='blue', bins = 30, alpha=0.5, label=f'Original (non-missing) = {len(df_numeric[col_name].dropna())}')
                ax.set_title(f'Dist. of {col_name} \n (SimpleImputer)', fontsize=10, fontweight = 'bold')
                ax.legend()
                ax.grid(alpha = 0.5)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig, width='content')   
            plt.close()
        st.write("### Green500 Imputation Comparison")
        col1, col2 = st.columns([1, 1]) 
        with col1:
            df_green = pd.read_csv("green500_combined.csv")
            df_green_combined_knn, df_green_numeric = perform_knn_imputation(df_green)

            columns_to_plot = ['Accelerator/Co-Processor Cores', 'Power (kW)']
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            for i, ax in enumerate(axs.flatten()):
                col_name = columns_to_plot[i]
                sns.histplot(df_green_combined_knn[col_name], ax=ax, kde=True, color='green', bins=30, alpha=0.5, label=f'Original + Imputed = {len(df_green_combined_knn[col_name])}')
                sns.histplot(df_green_numeric[col_name].dropna(), ax=ax, kde = True, color='blue', bins = 30, alpha=0.5, label=f'Original (non-missing) = {len(df_green_numeric[col_name].dropna())}')
                ax.set_title(f'Dist. of {col_name} \n (KNNImputer)', fontsize=10, fontweight = 'bold')
                ax.legend()
                ax.grid(alpha = 0.5)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig, width='content')  
            plt.close() 
        
        with col2:
            df_green = pd.read_csv("green500_combined.csv")
            df_green_combined_imputer, df_green_numeric_imputer = simple_imputer(df_green)

            columns_to_plot = ['Accelerator/Co-Processor Cores', 'Power (kW)']
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            for i, ax in enumerate(axs.flatten()):
                col_name = columns_to_plot[i]
                sns.histplot(df_green_combined_imputer[col_name], ax=ax, kde=True, color='green', bins=30, alpha=0.5, label=f'Original + Imputed = {len(df_green_combined_imputer[col_name])}')
                sns.histplot(df_green_numeric_imputer[col_name].dropna(), ax=ax, kde = True, color='blue', bins = 30, alpha=0.5, label=f'Original (non-missing) = {len(df_green_numeric_imputer[col_name].dropna())}')
                ax.set_title(f'Dist. of {col_name} \n (SimpleImputer)', fontsize=10, fontweight = 'bold')
                ax.legend()
                ax.grid(alpha = 0.5)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig, width='content')   
            plt.close()
            


        st.markdown(""" 
            **Observations:**
            - Simple Imputer strategy was easy and straightforward to implement. It did a good job of following the original distribution of the data, for most variables.
            - I checked against the KNN Imputer, and found that KNN tends to preserve the distribution of the data even better. This was in regards to the datacenters dataset.
            - The reason I beleived complex imputation like KNN is better than simple strategies is that the data we are imputing are actual countries. We know there is a big disparity between countries in terms of data center infrastructure, so it wouldn't be accurate to inflate numbers by using the median or mean, given the scale of the variables.
            - Finally, using KNN Imputer I saved the values to a `datacenters_imputed_df.csv` and `imputed_df.csv` file for further analysis.                 
            """)
        st.markdown("""---""")
        st.write("### Correlation Heatmaps")
        st.write("I then visualized a correlation heatmap to see how the imputed data correlates with other variables, and then compared it to the original.")
        st.write("As you can see there isn't a significant difference in correlation structure before and after imputation with KNN.")
        st.write("This can be from the fact that we had very little missingness to begin with, as well as KNN using similar values (neighbors).")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        data_og = pd.read_csv("data_centers_original.csv")
        corr_pre_imp = data_og.corr(numeric_only=True)
        sns.heatmap(corr_pre_imp, annot=True, linewidth=0.8, cmap="crest", fmt=".2f", ax=ax1)
        ax1.set_title("Heatmap before Imputation", fontweight='bold', fontsize=8) 

        data_post_imp = pd.read_csv("datacenters_imputed_df.csv")
        corr_post_imp = data_post_imp.corr(numeric_only=True)
        sns.heatmap(corr_post_imp, annot=True, linewidth=0.8, cmap="crest", fmt=".2f", ax=ax2)
        ax2.set_title("Heatmap after Imputation (KNN)", fontweight='bold', fontsize=8) 
        plt.tight_layout()
        st.pyplot(fig, width='content')   
        plt.close()
        
        st.write("Similarly for the Green500 dataset, I visualized correlation heatmaps before and after imputation.")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        green_og = pd.read_csv("green500_combined.csv")
        corr_green_pre_imp = green_og.corr(numeric_only=True)
        sns.heatmap(corr_green_pre_imp, annot=True, linewidth=0.8, cmap="crest", fmt=".2f", ax=ax1)
        ax1.set_title("Green500 Heatmap before Imputation", fontweight='bold', fontsize=8)

        green_post_imp = pd.read_csv("imputed_df.csv")
        corr_green_post_imp = green_post_imp.corr(numeric_only=True)
        sns.heatmap(corr_green_post_imp, annot=True, linewidth=0.8, cmap="crest", fmt=".2f", ax=ax2)
        ax2.set_title("Green500 Heatmap after Imputation (KNN)", fontweight='bold', fontsize=8)  

        plt.tight_layout()
        st.pyplot(fig, width='content')   
        plt.close()

    with st.expander("Transformations (Extra)", expanded=True):
        st.write("After imputation I wanted to check the distributions of the variables, and see if any transformations were necessary, mainly in reagrds to linear regression modeling.")
        st.write("After doing transformations I realized that the tradeoff between interpretability and normality wasn't worth it, so I decided to keep the variables in their original scale for now. This is specific to the datacenters dataset.")
        st.write("However, the next few plots show how the variables looked before and after transformations on this dataset.")
        col1, col2 = st.columns([1, 2])
        with col1:
            skewed_vars = []
            for col in df_numeric.columns:
                skewness = df_numeric[col].skew()
                if abs(skewness) > 1:
                    skewed_vars.append({
                        'Variable': col,
                        'Skewness': round(skewness, 2)
                    })

            skewed_df = pd.DataFrame(skewed_vars)
            st.write("Skewed variables in the dataset")
            st.dataframe(skewed_df, use_container_width=True, hide_index=True)
        with col2:
            st.write("I also wanted to explore how the distributions compared after standardizing the variables. So here's a simple plot.")
            scaler = StandardScaler()
            df_standardized = pd.DataFrame(scaler.fit_transform(df_numeric),columns=df_numeric.columns,index=df_numeric.index)

            fig, ax = plt.subplots(figsize=(12, 6))
            for col in ['total_data_centers', 'hyperscale_data_centers','colocation_data_centers', 'floor_space_sqft_total','power_capacity_MW_total', 'growth_rate_of_data_centers_percent_per_year']:
                sns.kdeplot(df_standardized[col], ax=ax, label=col)
            ax.set_xlabel('Standardized Value')
            ax.set_title('Comparing Distributions (Standardized)', fontweight='bold')
            ax.legend()
            st.pyplot(fig, width='content')
            plt.close()

        st.write("This was an additional exploration into how transformations can affect distributions of skewed variables, which I wanted to explore for my own understanding.")
        @st.fragment
        def transformation_plot_fragment():
            skewed_columns = ['total_data_centers', 'hyperscale_data_centers', 'colocation_data_centers', 
                            'floor_space_sqft_total', 'power_capacity_MW_total', 
                            'growth_rate_of_data_centers_percent_per_year']

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                selected_variable = st.selectbox("Select a variable to view transformations:", skewed_columns)

            fig = plot_transformations(df_numeric, selected_variable)

            col1, col2, col3 = st.columns([2, 0.5, 0.5])
            with col1:
                st.pyplot(fig, width='content')
            plt.close()

        transformation_plot_fragment()
    with st.expander("Merging Datasets", expanded = True):
        st.write("After cleaning both datasets, I merged them on the country column to create a final dataset for analysis.")
        st.markdown(""" 
            ### Process:
                    1. Read in both cleaned + imputed datasets.
                    2. Aggregated the Green500 dataset to country-level by calculating:
                        - Energy Efficiency [GFlops/Watts]: it's mean, median and max per country.
                        - Power (kW): mean and sum per country.
                        - Total Cores: mean and sum per country.
                        - Rmax (TFlops): mean and sum per country.
                        - Year of Installation: max and min for most recent and oldest supercomputers.
                        - Top500 Rank: A count per country
                        - Accelerator/Co-Processor Cores: mean per country.
                    3. Then saved these with appropriate column names.
                    4. Did a left join on the datacenters dataset, so the stats above would be added into that df.
                    5. Created an additional column classifying countries as 'Elite Supercomputing Nation' if a country was found in bothe datasets, else 'Infrastructure Only'.
                        - The logic behind this lies in that the Green500 dataset measured really the Elite nations using supercomputers, whereas the datacenters is ALL countries and more granular.
            """)
        st.markdown("""
            This final dataset was then saved as `merged_infrastructure_efficiency.csv` for further analysis. Snippet below:
            """)
        merged_df = pd.read_csv("merged_infrastructure_efficiency.csv")
        st.dataframe(merged_df.head(), width='stretch')
        
        with st.expander("Merged Data's Summary Statistics", expanded=True):
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

        st.subheader("Who Are the Elite Supercomputing Nations?")
        elite_nations = merged_df[merged_df['supercomputing_category'] == 'Elite Supercomputing Nation'][
            ['country', 'num_supercomputers', 'avg_energy_efficiency', 
             'total_data_centers', 'power_capacity_MW_total', 
             'average_renewable_energy_usage_percent']
        ].sort_values('num_supercomputers', ascending=False)
        
        st.dataframe(elite_nations, use_container_width=True, hide_index=True)