import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def show():
    st.title("Predictive Modeling")
    st.markdown("""
    Building a **Multiple Linear Regression Model** to modeling energy efficiency in supercomputing.
    Understanding what technological factors drive sustainable, high-performance computing. Specifically using the Green500 dataset as it has more observations and predictors.
    """)
    df_green = pd.read_csv("imputed_df.csv")
    
    with st.expander("Predicting Energy Efficiency (Green500)", expanded=True):
        st.subheader("Improved Multiple Linear Regression with Feature Engineering")
        st.markdown("""
        **Research Question**: What factors predict energy efficiency in the world's top supercomputers? 
        
        **Target Variable**: `Energy Efficiency [GFlops/Watts]` (Log-transformed for better fit)
        """)
        
        st.write("### Data Preparation & Feature Engineering")
        
        df_model = df_green[['Power (kW)', 'Total Cores', 'Year', 
                             'Processor Speed (MHz)', 'Accelerator/Co-Processor Cores',
                             'Energy Efficiency [GFlops/Watts]', 'Rmax [TFlop/s]']].copy()
        
        
        # Feature Engineering
        st.write("**Creating Engineered Features via transformations:**")
        
        # Log transformations
        df_model['log_power(kw)'] = np.log1p(df_model['Power (kW)'])
        df_model['log_total_cores'] = np.log1p(df_model['Total Cores'])
        df_model['log_accelerator_coprocessor_cores'] = np.log1p(df_model['Accelerator/Co-Processor Cores'])
        df_model['log_energy_efficiency'] = np.log1p(df_model['Energy Efficiency [GFlops/Watts]'])
        df_model['log_Rmax [TFlop/s]'] = np.log1p(df_model['Rmax [TFlop/s]'])
        
        
        feature_descriptions = pd.DataFrame({
            'Feature': ['log_power(kw)', 'log_total_cores', 'log_accelerator_coprocessor_cores', 'log_energy_efficiency', 'log_Rmax [TFlops/s]', 'has_accelerator'],
            'Description': [
                'Log of power',
                'Log of total cores',
                'Log of accelerator and co-processor cores',
                'Log of energy efficiency (our target variable)!',
                'Log of Rmax (Maximal LINPACK performance achieved)',
                'Whether a accelerator/processor exists for a supercomputer or not (binary), uses in Model 2'
            ]
        })
        st.dataframe(feature_descriptions, width='stretch', hide_index=True)
        
        # Model 1 Features
        st.write("### Model 1")
        features = ['Year', 'Processor Speed (MHz)', 'log_power(kw)',
                   'log_total_cores', 'log_accelerator_coprocessor_cores']
        target = 'log_energy_efficiency'
        
        X = df_model[features]
        y = df_model[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        st.write(f"**Train Set**: {len(X_train)} observations | **Test Set**: {len(X_test)} observations")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        st.write("### Model Training")
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Model performance
        st.write("### Model Performance")
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train R²", f"{train_r2:.4f}")
            st.metric("Test R²", f"{test_r2:.4f}")
        with col2:
            st.metric("Train RMSE", f"{train_rmse:.4f}")
            st.metric("Test RMSE", f"{test_rmse:.4f}")
        with col3:
            st.metric("Train MAE", f"{train_mae:.4f}")
            st.metric("Test MAE", f"{test_mae:.4f}")
        
        # Model coefficients
        st.write("### Model Coefficients (Standardized)")
        st.markdown("""
        These coefficients show the relative importance of each feature when all are on the same scale.
        """)
        
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.dataframe(coef_df[['Feature', 'Coefficient']], 
                        width='stretch', hide_index=True)

        
        with col2:
            residuals = y_test - y_pred_test
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred_test, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
            plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
            plt.xlabel('Predicted Values', fontsize=12)
            plt.ylabel('Residuals', fontsize=12)
            plt.title('Residual Plot: Checking for Patterns', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            st.pyplot(plt)
            plt.close()
#-------------------------------------------MODEL 2-------------------------------------------  
        st.write("### Model 2")
        st.markdown("""
                    Similarly instead of just using the 4 features used above (year, processor speed, log power, log total cores, log accelerator and coprocessor cores) to predict energy efficiency, I added one more.
                    - Added another variable from the df `has_accelerator` to check whether a specific supercomputer has one or not. 
                    - The results drastically improved!
                    """)
        
        df_green = pd.read_csv("imputed_df.csv")
        df_green['has_accelerator'] = df_green['Accelerator/Co-Processor'].notna().astype(int)
        df_green.loc[df_green['Accelerator/Co-Processor'].isna(), 'Accelerator/Co-Processor Cores'] = 0
        #print(df_green['has_accelerator'].value_counts())
        #print(df_green['Accelerator/Co-Processor Cores'].value_counts())
        df_model = df_green[['Power (kW)', 'Total Cores', 'Year', 
                                    'Processor Speed (MHz)', 'Accelerator/Co-Processor Cores',
                                    'Energy Efficiency [GFlops/Watts]', 'Rmax [TFlop/s]', 'has_accelerator']].copy()
        df_model['log_power(kw)'] = np.log1p(df_model['Power (kW)'])
        df_model['log_total_cores'] = np.log1p(df_model['Total Cores'])
        df_model['log_accelerator_coprocessor_cores'] = np.log1p(df_model['Accelerator/Co-Processor Cores'])
        df_model['log_energy_efficiency'] = np.log1p(df_model['Energy Efficiency [GFlops/Watts]'])
        df_model['log_Rmax [TFlop/s]'] = np.log1p(df_model['Rmax [TFlop/s]'])
        features = ['Year', 'Processor Speed (MHz)', 'log_power(kw)',
                        'log_total_cores', 'log_accelerator_coprocessor_cores', 'has_accelerator']
        target = 'log_energy_efficiency'
        X = df_model[features]
        y = df_model[target]
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        st.write(f"**Train Set**: {len(X_train)} observations | **Test Set**: {len(X_test)} observations")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        st.write("### Model Training")
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Model performance
        st.write("### Model Performance")
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train R²", f"{train_r2:.4f}")
            st.metric("Test R²", f"{test_r2:.4f}")
        with col2:
            st.metric("Train RMSE", f"{train_rmse:.4f}")
            st.metric("Test RMSE", f"{test_rmse:.4f}")
        with col3:
            st.metric("Train MAE", f"{train_mae:.4f}")
            st.metric("Test MAE", f"{test_mae:.4f}")
        
        # Model coefficients
        st.write("### Model Coefficients (Standardized)")
        
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.dataframe(coef_df[['Feature', 'Coefficient']], 
                        width='stretch', hide_index=True)
        
        with col2:
            residuals = y_test - y_pred_test
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred_test, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
            plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
            plt.xlabel('Predicted Values', fontsize=12)
            plt.ylabel('Residuals', fontsize=12)
            plt.title('Residual Plot', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            st.pyplot(plt)
            plt.close()
        with col3:
            plt.figure(figsize=(10, 6))
            plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='darkgreen')
            plt.xlabel('Residuals', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
            plt.grid(alpha=0.3)
            st.pyplot(plt )
            plt.close
#----------------------------------FINAL TAKEAWAY---------------------------------------------------#
        # Model interpretation
        st.write("### Model Interpretation")
        
        # Find most impactful features
        coef_sorted = coef_df.sort_values('Coefficient', ascending=False)
        most_positive = coef_sorted.iloc[0]
        most_negative = coef_sorted.iloc[-1]
        
        st.markdown(f"""
        **Key Insights from the Improved Model:**
        
        1. **Model Fit**: The log-transformed model 2 explains **{test_r2*100:.2f}%** of variance 
           (R² = {test_r2:.4f}).
        
        2. **Most Positive Impact from Model 2**: `{most_positive['Feature']}` (coefficient: {most_positive['Coefficient']:.4f})
           - This suggests that the year a supercomputer is installed is one of the strongest predictors of efficiency.
        
        3. **Most Negative Impact**: `{most_negative['Feature']}` (coefficient: {most_negative['Coefficient']:.4f})
           - This makes sense as higher power consumption can lead to lower efficiency scores.
        
        4. **Residual Behavior**: Log transformation doesn't have patterns that are shown in the residuals.
        
        5. **No Negative Predictions**: Log transformation ensures all predictions are positive and realistic.
        """)
        
        # Additional insights
        st.write("### Why This Model Works Better")
        st.markdown("""
        **Technical Improvements:**
        
        1. **Log Transformation**: Energy efficiency, power, and cores are naturally multiplicative 
           (doubling cores doesn't linearly double efficiency). Log transformation captures this.
        
        2. **Standardization**: Puts all features on same scale, making coefficients directly comparable.
        
        4. **Reduced Heteroscedasticity**: Log transformation stabilizes variance across prediction range.
        
        5. **Stopping at current variables?**: I stopped at an R² of 0.83 because it works with real-world data. Real systems have noise and the perfect prediction wouldn't exist. If I wanted a R² of 0.90, it could require more than 10 features which hurt interpretability and can risk overfitting. We can explain most of the variance with 6 features.
        
        **Takeaways:**
        - Model reveals that efficiency gains come from **smart architecture** (accelerators), as well as timing
          more than raw scale like power.
                    
        - **Newer systems** consistently more efficient → technological progress is working. It might be worth upgrading to newer system if you want efficiency gains and a sustainable planet.
        
        - Understanding these patterns helps design future data centers that are **more energy efficient**.
        """)