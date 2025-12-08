import streamlit as st

def show():
    st.markdown("""
                ### The Two-Tier Reality of Global Computing

                **The Green500 Dataset** captures the world's most energy-efficient supercomputers — 
                    representing nations at the **pinnacle of computational achievement**. Only ~30-35 countries 
                    appear in these rankings.

                **The Data Centers Dataset** provides a **comprehensive view of 191 countries**, showing 
                infrastructure investment regardless of supercomputing presence.

                **The Gap**: Many countries are building extensive data center infrastructure (colocation, 
                hyperscale facilities, power capacity) but don't appear in global supercomputing rankings. 

                **Key Questions**:
                1. **Among the Elite**: What makes supercomputers in top-performing countries more efficient?
   
                2. **The Broader Question**: What infrastructure characteristics distinguish countries that 
                achieve supercomputing excellence from those that don't? (Classification Model on merged data)

                This two-tiered approach reveals both the technological factors driving efficiency AND the 
                infrastructure prerequisites for entering the global supercomputing arena.
                """)
    
    # Dataset 1
    st.markdown("### 1. Data Center Infrastructure Dataset")
    st.markdown("""
    Country-level insights into global data center infrastructure (as of 2025):
    
    **Key Features:**
    - Data center counts (total, hyperscale, colocation).
    - Power capacity in megawatts (MW).
    - Average renewable energy usage (%).
    - Average growth rates.
    - Internet Penertration (%).
    """)
    st.link_button(url="https://www.kaggle.com/datasets/rockyt07/data-center-dataset/data", 
                 label="Data Center Dataset", icon="🏢")
    
    st.markdown("---")
    
    # Dataset 2
    st.markdown("### 2. Green500")
    st.markdown("""
    In the Green500 the systems are ranked by how much computational performance they deliver on the HPL benchmark per Watt of electrical power consumed.
    
    Data range spans from 2020 - 2025, chosen due to the boom in AI over the past few years, and the need for supercomputing resources to train large language models.
    \n\n The Green500 list is updated twice a year in June and November. Data was only chosen from November lists to ensure consistency and reduce redundancy.
    
    **Key Features:**
    - 1,161 rows x 19 columns (post-imputation).
    - Key Infrastructure metrics: number of cores.
    - Other indicators: power, energy efficiency, processor speed, installation year.
    
    """)
    st.link_button(label="Green500 Dataset", url="https://top500.org/lists/green500/", icon="🌿")