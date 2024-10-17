import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
# import ipywidgets as widgets


def main():
    data = st.file_uploader("📁 Load Data File", type=["csv", "xlsx"])
    if data is not None:
        df = pd.read_csv(data)
        st.write(df)
        columns = list(df.columns)
        viz = ['bar 📊', 'pie 🥧', 'correlation 🎢', 'Line 🪡', 'Trend 📈']
        tabs = st.tabs(viz)
        with tabs[0]:
            if 'Year' in columns:
                df['Year'] = pd.to_datetime(df['Year'], format='%Y')
                numeric_columns = df.select_dtypes(include='number').columns.tolist()
                string_columns = df.select_dtypes(include='object').columns.tolist()
                param_1 = st.selectbox("Select X value" ,string_columns, placeholder="Select one parameter", index=None)
                param_2 = st.selectbox("Select Y value" ,numeric_columns, placeholder="Select one parameter", index=None)
                filter = st.selectbox("Select filter",string_columns, placeholder="Select one parameter", index=None)
                min_year = df['Year'].min().year
                max_year = df['Year'].max().year
                year_range = st.slider('Seleccione un rango de años:',
                                    min_value=min_year, 
                                    max_value=max_year, 
                                    value=(min_year, max_year))
                df_filtered = df[(df['Year'].dt.year >= year_range[0]) & (df['Year'].dt.year <= year_range[1])]
                if param_1 and param_2 or filter:
                    fig, ax = plt.subplots()
                    sns.barplot(data=df_filtered, x=param_1, y=param_2, hue=filter, ax=ax, errorbar=None)
                    ax.set_title(f'Total {param_2} by {param_1} filtering {filter}', fontsize=14)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    st.pyplot(fig)
                else:
                    st.warning("Please Select Your Variable")
            else:
                st.warning("Por favor, carga un archivo CSV o XLSX.")
        
        with tabs[1]:
            st.write('hul;a')
        
# def main():
#     if 'data' not in st.session_state:
#         param_1 = ''
        
#     data = st.file_uploader("📁 Load Data File", type=["csv", "xlsx",])
#     if data is not None:
#         df=pd.read_csv(data)
#         st.write(df)
#         columns = list(df.columns)
#         # st.write(columns)
#         viz = ['bar 📊','pie 🥧', 'correlation 🎢','Line 🪡','Trend 📈']
#         tabs = st.tabs(viz)
#         # year =  widgets.IntRangeSlider(value=[min(data['Year']), max(data['Year'])],
#         #                                min=df['Año'].min(),max=df['Año'].max(),
#         #                                step=1,description='Year Ranges',continuous_update=False)
#         param_1 = st.selectbox("Select X value", columns,placeholder="Select one parameter",index=None)
#         param_2 = st.selectbox("Select Y value", columns,placeholder="Select one parameter",index=None)
#         filter = st.selectbox("Select filter", columns,placeholder="Select one parameter",index=None)
#         df['Year'] = pd.to_datetime(df['Year'], format='%Y')
        
#         if st.button('Create Barplot'):
#             min_year = df['Year'].min().year
#             max_year = df['Year'].max().year
#             year_range = st.slider('Seleccione un rango de años:',
#                                    min_value=min_year, 
#                                    max_value=max_year, 
#                                    value=(min_year, max_year))
#             df_filtered = df[(df['Year'].dt.year >= year_range[0]) & (df['Year'].dt.year <= year_range[1])]
#             if param_1 and param_2 or filter:                   
#                 fig, ax = plt.subplots()
#                 sns.barplot(df, x=param_1, y=param_2,hue=filter,ax=ax,errorbar=None)
#                 ax.set_title(f'Total {param_2} by {param_1} filtering {filter}', fontsize=14)
#                 ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#                 st.pyplot(fig)
            # else: 
                
        # with col1:
        #     with tabs[0]:
        #         st.header('Barplots')
        #         st.write('Aquí es donde puedes preprocesar tus datos.')
        #         param_1 = st.selectbox("Select Parameters", columns,placeholder="Select one parameter",index=0,)
                
        # with col2:
        #     with tabs[1]:
        #         st.header('Barplots')
        #         st.write('Aquí es donde puedes preprocesar tus datos.')
        #         param_2 = st.selectbox("Select Parameters", columns,placeholder="Select one parameter",index=1,)
        # with col3:
        #     with tabs[2]:
        #         st.header('Barplots')
        #         st.write('Aquí es donde puedes preprocesar tus datos.')
        #         filter = st.selectbox("Select Parameters", columns,placeholder="Select one parameter",index=2,)

            # 
            
            # if filter is not None:
            #     fig = px.bar(df, x=param_1, y=param_2, color='Genre', barmode='group',hue=filter)
            # else:
            #     fig = px.bar(df, x=param_1, y=param_2, color='Genre', barmode='group')
                
            # st.plotly_chart(fig)
        
    # Agregar contenido a la pestaña de Entrenamiento del modelo
    #     with tabs[1]:
    #         st.header('Entrenamiento del modelo')
    #         st.write('Aquí es donde puedes entrenar tu modelo.')
        
    #     # Agregar contenido a la pestaña de Evaluación del modelo
    #     with tabs[2]:
    #         st.header('Evaluación del modelo')
    #         st.write('Aquí es donde puedes evaluar tu modelo.')
        
    #     # Agregar contenido a la pestaña de Visualización de resultados
    #     with tabs[3]:
    #         st.header('Visualización de resultados')
    #         st.write('Aquí es donde puedes visualizar tus resultados.')
        
        
    #     # fig = px.bar(df, x='Platform', y='NA_Sales', color='Genre', barmode='group')
    #     # st.plotly_chart(fig)
    # else:
    #     st.warning("No File uploaded.")
    

if __name__ == "__main__":
    main()