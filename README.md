# Sustainable Supercomputing & Data Center Infrastructure

An interactive Streamlit dashboard exploring the intersection of energy-efficient computing, supercomputer performance, and global data center sustainability.

## Project Overview

This project analyzes two complementary datasets to understand how modern supercomputers and data centers balance computational power with environmental responsibility. With the explosive growth of AI and machine learning, understanding energy efficiency in high-performance computing is important.

## Key Objectives

- Explore global data center infrastructure and renewable energy adoption
- Analyze energy efficiency trends in supercomputing (2020-2025)
- Build predictive models to identify factors driving computational sustainability
- Provide insights for policymakers and researchers on green computing strategies

## Datasets

### 1. Data Center Infrastructure Dataset
**Source:** [Kaggle - Data Center Dataset](https://www.kaggle.com/datasets/rockyt07/data-center-dataset/data)

Country-level insights into global data center infrastructure (191 countries):
- Data center counts (total, hyperscale, colocation)
- Power capacity (MW) and floor space
- Renewable energy usage percentages
- Internet penetration and growth rates

### 2. Green500 Dataset
**Source:** [Top500 Green500 Lists](https://top500.org/lists/green500/)

Supercomputer energy efficiency rankings (1,159 systems, 2020-2025):
- Energy efficiency metrics (GFlops/Watt)
- Hardware specifications (cores, accelerators, power consumption)
- Performance benchmarks (LINPACK Rmax/Rpeak)
- Temporal trends in green computing

**Note:** Data collected from November lists only (2020-2025) to ensure consistency during the AI computing boom.

## 🛠️ Project Structure (for streamlit)

```
├── app.py                          # Main Streamlit application
├── Cleaning.py                     # Data cleaning and imputation
├── Documentation.py                # Dataset documentation
├── EDA.py                          # Exploratory data analysis
├── Modeling.py                     # Linear regression modeling
```
Other dataframes exist in the root directory so file paths are compatible when streamlit runs the code. If you wanted to see what the code looks like in regards to each dataset, it is organized by name. The analysis post merging is also in the root directory. 


## Running the Dashboard

### Prerequisites
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
```

### Launch Application
```bash
streamlit run app.py
```

### Navigation
- **Documentation:** Dataset overviews and sources
- **Cleaning:** Imputation strategies and transformation analysis + extras
- **EDA:** Interactive visualizations of both datasets, and merged df
- **Modeling:** Linear regression results and interpretations

As AI models grow larger, these insights become critical for responsible computing. Training a large language model on modern accelerated hardware isn't just faster, it's more environmentally responsible (or atleast it should be).

## Future Work
- Incorporate real-time energy pricing data
- Add cooling technology analysis (liquid vs. air)
- Expand to include carbon footprint calculations
- Build recommendation system for optimal hardware configurations


