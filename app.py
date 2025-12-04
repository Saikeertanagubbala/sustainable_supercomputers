import streamlit as st

from pages import EDA, Cleaning, Documentation, Modeling

st.set_page_config(page_title="Supercomputing", layout="wide")

st.title("🌎 Sustainable Supercomputing & Data-Center Infrastructure")
st.write("Gaining a better understanding on **energy-efficient computing**, exploring how modern **supercomputers** and **global data centers** shape sustainability, environmental impact, and responsible digital infrastructure.")
st.markdown("""
    <style>
    .stButton button {
            background-color: #287AB8;
            width: 100%;
            border-radius:0.5rem;
            padding: 0.5rem 1rem;
    }
    
    .stButton button:hover {
            background-color: #769BB8;
    }
    
    .active-button {
            background-color: #58BC82 !important;
            color: black !important;
            border-radius:0.5rem !important;
            padding: 0.5rem 1rem !important;
            text-align: center !important;
            width: 100% !important;
            display: block !important;
    }
    
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
    if st.session_state.page == "Documentation":
        st.markdown('<div class="active-button">Documentation</div>', unsafe_allow_html=True)
    else:
        if st.button("Documentation", use_container_width=True):
            st.session_state.page = "Documentation"
            st.rerun()
with col2:
    if st.session_state.page == "Cleaning":
        st.markdown('<div class="active-button">Cleaning</div>', unsafe_allow_html=True)
    else:
        if st.button("Cleaning", use_container_width=True):
            st.session_state.page = "Cleaning"
            st.rerun()
with col3:
    if st.session_state.page == "EDA":
        st.markdown('<div class="active-button">EDA</div>', unsafe_allow_html=True)
    else:
        if st.button("EDA", use_container_width=True):
            st.session_state.page = "EDA"
            st.rerun()

with col4:
    if st.session_state.page == "Modeling":
        st.markdown('<div class="active-button">Modeling</div>', unsafe_allow_html=True)
    else:
        if st.button("Modeling", use_container_width=True):
            st.session_state.page = "Modeling"
            st.rerun()

# Displaying pages
if st.session_state.page == "Documentation":
    Documentation.show()
elif st.session_state.page == "Cleaning":
    Cleaning.show()
elif st.session_state.page == "EDA":
    EDA.show()
elif st.session_state.page == "Modeling":
    Modeling.show()