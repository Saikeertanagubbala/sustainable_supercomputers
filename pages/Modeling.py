import streamlit as st

def show():
    st.write("Multi-Linear Regression and Model Evaluation plots will be displayed here.")
    st.markdown("""
    **Target Variable:** Energy Efficiency (Gflops/Watt)
    Predictors used in the 1st itertaion of the model were simple continuous variables such as: 'Year', 'log_total_cores', 'log_rmax', 'log_power', 'Processor Speed (MHz)'
    """)
    st.write("More modeling with different variables need to be done to ensure better accuracy and performance of the model.")