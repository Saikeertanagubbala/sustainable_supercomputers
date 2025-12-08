import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("datacenters_imputed_df.csv")
df_merged = pd.read_csv("merged_infrastructure_efficiency.csv")
df_green = pd.read_csv("imputed_df.csv")

def show():
    with st.expander("EDA Insights", expanded=True):
        st.markdown("""
        Using the merged dataset, several insights can be taken away from the visualizations below:
        1. Below we see a clear distintion between countries that are elite and those that just have the infrastructure. Elite supercomputing nations tend to have both higher power capacity and larger floor space for their data centers.
        2. There is a positive correlation between power capacity and floor space of data centers, indicating that larger data centers tend to have higher power capacities.
        3. The United States, China and Germany stand out as countries with both high power capacity and large floor space, indicating their significant investment in data center infrastructure.  
                """)
        fig = px.scatter(df_merged, x='floor_space_sqft_total', y='power_capacity_MW_total', hover_name='country', log_x=True, log_y=True, hover_data=['total_data_centers'], color='supercomputing_category',
                 color_discrete_map={'Infrastructure Only': '#FF6B6B', 'Elite Supercomputing Nation': "#48B629"})
    

        fig.update_layout(
            title = dict(
                text='Power Capacity vs. Floor Space of Global Data Centers', #actual text
                font = dict(color="#ECECEB", size=22), #color +size of title
                x=0.35, #position of title
            ),
            xaxis_title=dict(
                text="Floor Space (sqft)", #xaxis title
                font=dict(size=18) #xaxis title font size
            ),
            yaxis_title = dict(
                text="Power Capacity (MW)",
                font=dict(size=18)
            ),
            legend=dict(
            font=dict(size=16), 
            title=dict(text="Supercomputing Category", font=dict(size=18))
            ),
            
            height=600,
            width=900,
            
            yaxis=dict( #yaxis ticks specs
                type="log",
                dtick=1,
                color = "#FFFFFF",
                linewidth=1,
                linecolor='white',
                tickfont=dict(size=16)
            ),
            xaxis=dict(
                color="#FFFFFF",
                linewidth=1,
                linecolor='white',
                tickfont=dict(size=16)),
            
            yaxis_gridcolor='gray',
        )

        fig.update_traces(
            marker=dict(
            size=8)
            )
        st.plotly_chart(fig, config = {'width':'stretch'})

#--------------------------------2nd plot----------------------------------------------------------------------------------------------------#
        st.markdown("""
                This plot explore power capacity and renewable energy usage. Key insights:
                1. Seems to be a trend that the countries who consume more power tend to use less renewable energy, indicating a potential trade-off between power capacity and renewable energy adoption. This is apparent with the US, which is sticking out key point on the x-axis of power capacity but ranks low on renewable energy.
                2. Infact we see more infrasturcture only countries not only use less power capacity but also have higher renewable energy usage, indicating that these countries may be focusing on sustainable growth of their data center infrastructure.
                3. Countries like Iceland stand out as they have both moderate power capacity and high renewable energy usage, showcasing successful integration of sustainability in their data center operations.
                    """)
        fig = px.scatter(df_merged, 
         x='power_capacity_MW_total', 
         y='average_renewable_energy_usage_percent',
         hover_name='country',
         hover_data=['total_data_centers', 'floor_space_sqft_total'],
         log_x=True,
         color='supercomputing_category',
         marginal_x="violin",
         marginal_y="box")

        fig.update_layout(
            height=600,
            width=900,
            title=dict(
                text='Relationship b/w Power Capacity & Renewable Energy Usage',
                font=dict(color="#F8F9F8", size=22),
                x=0.35
            ),
            xaxis_title=dict(
                text="Total Power Capacity (MW)",
                font=dict(size=18, color="#FFFFFF")
            ),
            yaxis_title=dict(
                text="Average Renewable Energy (%)",
                font=dict(size=18, color="#FFFFFF")
            ),
            xaxis=dict(
                dtick=1,
                gridcolor='rgba(128,128,128,0.5)',
                linewidth=1,
                linecolor='white'
            ),
            yaxis=dict(
                gridcolor='rgba(128,128,128,0.5)',
                linewidth=1,
                linecolor='white'
            ),
            legend=dict(
            font=dict(size=16), 
            title=dict(text="Supercomputing Category", font=dict(size=18))
            ),
            yaxis_gridcolor='gray',
        )

        fig.update_traces(
            marker=dict(
                size=8,
                line=dict(width=1, color='#2D5016')
            ),
            selector=dict(type='scatter')
        )

        st.plotly_chart(fig, config = {'width':'stretch'})
#-----------------------------------------3rd plot-------------------------------------------------------------------------------------------#
        st.markdown("""
                    This plot focuses on the datacenters dataset (not the merged df), and shows countries split into regions and how they vary by certain metrics listed in the dataset to see some trends.
                    - If you zoom into specifically the renewable energy and internet penetration axis you can see a strong-positive linear trend between the two.
                    """)
        fig = px.scatter_3d(df, x='average_renewable_energy_usage_percent', 
                            y='growth_rate_of_data_centers_percent_per_year', 
                            z='internet_penetration_percent',
                            color='region',
                            opacity=0.7,
                            hover_name='country')

        fig.update_layout(
            height=600, 
            width=700,
            title=dict(
                text='Countries measured with different factors',
                font=dict(color="#F8F9F8", size=22),
                x=0.3
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(
                        text="Avg Renewable Energy Usage (%)",
                        font=dict(size=18, color="#FFFFFF")
                    ),
                    gridcolor='rgba(128,128,128,0.5)'
                ),
                yaxis=dict(
                    title=dict(
                        text="Growth Rate (% per year)",
                        font=dict(size=18, color="#FFFFFF")
                    ),
                    gridcolor='rgba(128,128,128,0.5)'
                ),
                zaxis=dict(
                    title=dict(
                        text="Internet Penetration (%)",
                        font=dict(size=18, color="#FFFFFF")
                    ),
                    gridcolor='rgba(128,128,128,0.5)'
                )
            ),
            legend=dict(
                font=dict(size=16),  
                title=dict(
                    text='Region',
                    font=dict(size=18)
                )
            )
        )

        fig.update_traces(
            marker=dict(
                size=6,
                line=dict(width=1)
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("EDA specifically for Green500", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            The government and research sector take up majority of the supercomputer usage, atleast within the past 5 years.
            """)

            result = df_green.groupby('Segment')['Power (kW)'].mean()
            percentages = (result / result.sum())*100
            names = result.index
            size = percentages
            plt.figure(figsize=(8,6))
            my_circle = plt.Circle( (0,0), 0.7, color='white')
            colors = ['#287AB8']
            explode = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
            plt.pie(size, labels=names, colors=colors, autopct='%1.1f%%', pctdistance=0.85, explode=explode)
            p = plt.gcf()
            p.gca().add_artist(my_circle)
            plt.title("Average Energy Demand by Supercomputer Application Sector", fontweight='bold')
            st.pyplot(plt, width='content')
            plt.close()
#---------------------------------2nd plot----------------------------------------------------------------------------------------------------#
        with col2:
            st.write("Energy efficiency has increased over time from when the supercomputers have been installed.")
            avg_energy_efficiency_per_year = df_green.groupby('Year')['Energy Efficiency [GFlops/Watts]'].mean()

            plt.scatter(avg_energy_efficiency_per_year.index,avg_energy_efficiency_per_year.values,color='darkgreen',alpha=0.7, edgecolors='black')

            plt.xlabel('Year')
            plt.ylabel('Energy Efficiency (GFlops/Watts)')
            plt.title('Average Energy Efficiency by Installation Year')
            plt.grid(alpha=0.4)
            st.pyplot(plt, width='content')
            plt.close()

#---------------------------------3rd plot----------------------------------------------------------------------------------------------------#
        with col3:
            st.write("It seems as power increases energy efficiency decreases.")
            plt.scatter(df_green["Power (kW)"],df_green["Energy Efficiency [GFlops/Watts]"], color='darkgreen',alpha=0.7, edgecolors='black')
            plt.title('Trends in Increase of Power with Energy Efficiency')
            plt.grid(alpha=0.4)
            plt.xscale('log')
            plt.xlabel("Log Power")
            plt.ylabel("Energy Efficiency (GFlops/Watts)")
            st.pyplot(plt, width='content')
            plt.close()