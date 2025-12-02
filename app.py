import streamlit as st

from pages import Documentation, analysis, results, Modeling

st.set_page_config(page_title="Supercomputing", layout="wide")

st.title("🌎 Sustainable Supercomputing & Data-Center Infrastructure")
st.write("Gaining a better understanding on **energy-efficient computing**, exploring how modern **supercomputers** and **global data centers** shape sustainability, environmental impact, and responsible digital infrastructure.")
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)
if "page" not in st.session_state:
    st.session_state.page = "Documentation"

# Navigation
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Documentation", use_container_width=True):
        st.session_state.page = "Documentation"
        st.rerun()
with col2:
    if st.button("Analysis", use_container_width=True):
        st.session_state.page = "analysis"
        st.rerun()
with col3:
    if st.button("Results", use_container_width=True):
        st.session_state.page = "results"
        st.rerun()

with col4:
    if st.button("Modeling", use_container_width=True):
        st.session_state.page = "Modeling"
        st.rerun()

# Displaying pages
if st.session_state.page == "Documentation":
    Documentation.show()
elif st.session_state.page == "analysis":
    analysis.show()
elif st.session_state.page == "results":
    results.show()
elif st.session_state.page == "Modeling":
    Modeling.show()