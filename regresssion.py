import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  
# import ipywidgets as widgets

def main():
    data = st.file_uploader("📁 Load Data File", type=["csv", "xlsx"])
    if data is not None:
        df = pd.read_csv(data)
        st.write(df)
        columns = list(df.columns)
        viz = ['bar 📊', 'pie 🥧', 'correlation 🎢', 'Line 🪡', 'Trend 📈']
        tabs = st.tabs(viz)
        
        # Selección del campo de fecha (puede ser "Year", "year", "Date", "date")
        date_columns = [col for col in columns if 'year' in col.lower() or 'date' in col.lower()]
        
        if len(date_columns) == 0:
            st.warning("No Date or Year column found")
            return
        
        with tabs[0]:
            date_field = st.selectbox("Select Date Field for Bar Plot", date_columns)
            if date_field:
                # Si el campo contiene 'year', tratarlo como año
                if 'year' in date_field.lower():
                    df[date_field] = pd.to_numeric(df[date_field], errors='coerce')  # Asegurar que sea numérico
                    df = df.dropna(subset=[date_field])  # Eliminar valores NaN
                    min_year = int(df[date_field].min())
                    max_year = int(df[date_field].max())
                    
                    # Slider para el rango de años
                    year_range = st.slider('Select Year range for Bar Plot',
                                           min_value=min_year, 
                                           max_value=max_year, 
                                           value=(min_year, max_year))
                    
                    # Filtrar los datos por el rango seleccionado
                    df_filtered = df[(df[date_field] >= year_range[0]) & (df[date_field] <= year_range[1])]
                    
                # Si el campo contiene 'date', tratarlo como fecha completa
                elif 'date' in date_field.lower():
                    df[date_field] = pd.to_datetime(df[date_field], errors='coerce')  # Convertir a fecha
                    df = df.dropna(subset=[date_field])  # Eliminar valores NaT
                    min_date = df[date_field].min().date()
                    max_date = df[date_field].max().date()
                    
                    # Slider para el rango de fechas
                    year_range = st.slider('Select Date range for Bar Plot',
                                           min_value=min_date, 
                                           max_value=max_date, 
                                           value=(min_date, max_date))
                    
                    # Filtrar los datos por el rango seleccionado
                    df_filtered = df[(df[date_field].dt.date >= year_range[0]) & (df[date_field].dt.date <= year_range[1])]
                
                # Selección de las variables para el gráfico
                numeric_columns = df_filtered.select_dtypes(include='number').columns.tolist()
                string_columns = df_filtered.select_dtypes(include='object').columns.tolist()
                
                param_1 = st.selectbox("Select X value for Barplot", string_columns, placeholder="Select one parameter",index=None)
                param_2 = st.selectbox("Select Y value for Barplot", numeric_columns, placeholder="Select one parameter",index=None)
                filter = st.selectbox("Select filter for Barplot", string_columns, placeholder="Select one parameter",index=None)
                
                if param_1 and param_2 and filter:
                    fig, ax = plt.subplots()
                    sns.barplot(data=df_filtered, x=param_1, y=param_2, hue=filter, ax=ax, errorbar=None)
                    ax.set_title(f'Total {param_2} by {param_1} filtering {filter}', fontsize=14)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    st.pyplot(fig)
                else:
                    st.warning("Please Select Your Variables")
            else:
                st.warning("Por favor, selecciona un campo de fecha válido.")

        # Pestaña de pie chart
        with tabs[1]:
            date_field_pie = st.selectbox("Select Date Field for Pie Chart", date_columns)
            if date_field_pie:
                # Tratar el campo como año o fecha, según su tipo
                if 'year' in date_field_pie.lower():
                    df[date_field_pie] = pd.to_numeric(df[date_field_pie], errors='coerce')
                    df = df.dropna(subset=[date_field_pie])
                    min_year = int(df[date_field_pie].min())
                    max_year = int(df[date_field_pie].max())
                    
                    # Slider para el rango de años
                    year_range_2 = st.slider('Select Year range for Pie Chart',
                                             min_value=min_year, 
                                             max_value=max_year, 
                                             value=(min_year, max_year))
                    
                    df_filtered = df[(df[date_field_pie] >= year_range_2[0]) & (df[date_field_pie] <= year_range_2[1])]
                    
                elif 'date' in date_field_pie.lower():
                    df[date_field_pie] = pd.to_datetime(df[date_field_pie], errors='coerce')
                    df = df.dropna(subset=[date_field_pie])
                    min_date = df[date_field_pie].min().date()
                    max_date = df[date_field_pie].max().date()
                    
                    # Slider para el rango de fechas
                    year_range_2 = st.slider('Select Date range for Pie Chart',
                                             min_value=min_date, 
                                             max_value=max_date, 
                                             value=(min_date, max_date))
                    
                    df_filtered = df[(df[date_field_pie].dt.date >= year_range_2[0]) & (df[date_field_pie].dt.date <= year_range_2[1])]
                
                # Selección de variables para el gráfico de pie
                numeric_columns_2 = df_filtered.select_dtypes(include='number').columns.tolist()
                string_columns_2 = df_filtered.select_dtypes(include='object').columns.tolist()
                
                param_3 = st.selectbox("Select X value for PieChart", string_columns_2, placeholder="Select one parameter", index=None)
                param_4 = st.selectbox("Select Y value for PieChart", numeric_columns_2, placeholder="Select one parameter", index=None)
                
                if param_3 and param_4:
                    df_2 = df_filtered[[param_3, param_4]]
                    df_sum = df_2.groupby(param_3).sum().reset_index()
                    df_top10 = df_sum.nlargest(10, param_4)
                    a = list(df_top10[param_4])
                    b = list(df_top10[param_3])
                    
                    if param_3 and param_4:
                        fig, ax = plt.subplots()
                        plt.pie(a, labels=b, autopct='%1.1f%%', labeldistance=1.2,
                                pctdistance=0.85,wedgeprops={'linewidth': 1, 'edgecolor': 'black'})
                        ax.set_title(f'{param_3} distribution by {param_4}', fontsize=14)
                        st.pyplot(fig)
                    else:
                        st.warning("Please Select Your Variables")
                else:
                    st.warning("Por favor, selecciona un campo de fecha válido.")



# def main():
#     data = st.file_uploader("📁 Load Data File", type=["csv", "xlsx"])
#     if data is not None:
#         df = pd.read_csv(data)
#         st.write(df)
#         columns = list(df.columns)
#         viz = ['bar 📊', 'pie 🥧', 'correlation 🎢', 'Line 🪡', 'Trend 📈']
#         tabs = st.tabs(viz)
#         with tabs[0]:
            
#             if 'Year' in columns or 'date' in columns or 'Date' in columns:
#                 df['Year'] = pd.to_datetime(df['Year'], format='%Y')
#                 numeric_columns = df.select_dtypes(include='number').columns.tolist()
#                 string_columns = df.select_dtypes(include='object').columns.tolist()
#                 param_1 = st.selectbox("Select X value for Barplot" ,string_columns, placeholder="Select one parameter", index=None)
#                 param_2 = st.selectbox("Select Y value for Barplot" ,numeric_columns, placeholder="Select one parameter", index=None)
#                 filter = st.selectbox("Select filter",string_columns, placeholder="Select one parameter", index=None)
#                 min_year = df['Year'].min().year
#                 max_year = df['Year'].max().year
#                 year_range = st.slider('Select Year range for BarPlot',
#                                     min_value=min_year, 
#                                     max_value=max_year, 
#                                     value=(min_year, max_year))
#                 df_filtered = df[(df['Year'].dt.year >= year_range[0]) & (df['Year'].dt.year <= year_range[1])]
#                 if param_1 and param_2 or filter:
#                     fig, ax = plt.subplots()
#                     sns.barplot(data=df_filtered, x=param_1, y=param_2, hue=filter, ax=ax, errorbar=None)
#                     ax.set_title(f'Total {param_2} by {param_1} filtering {filter}', fontsize=14)
#                     ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#                     st.pyplot(fig)
#                 else:
#                     st.warning("Please Select Your Variables")
#             else:
#                 st.warning("Por favor, carga un archivo CSV o XLSX.")
        
#         with tabs[1]:
#             if 'Year' in columns:
#                 df['Year'] = pd.to_datetime(df['Year'], format='%Y')
#                 numeric_columns_2 = df.select_dtypes(include='number').columns.tolist()
#                 string_columns_2 = df.select_dtypes(include='object').columns.tolist()
#                 param_3 = st.selectbox("Select X value for PieChart" ,string_columns_2, placeholder="Select one parameter", index=None)
#                 param_4 = st.selectbox("Select Y value for PieChart" ,numeric_columns_2, placeholder="Select one parameter", index=None)
#                 if param_3 and param_4:
#                     df_2 = df[[param_3,param_4]]
#                     df_sum = df_2.groupby(param_3).sum().reset_index()
#                     a = list(df_sum[param_4])
#                     b = list(df_sum[param_3])
#                     min_year = df['Year'].min().year
#                     max_year = df['Year'].max().year
#                     year_range_2 = st.slider('Select Year range for Pie Chart',
#                                         min_value=min_year, 
#                                         max_value=max_year, 
#                                         value=(min_year, max_year))
#                     df_filtered = df[(df['Year'].dt.year >= year_range_2[0]) & (df['Year'].dt.year <= year_range[1])]
#                     if param_3 and param_4:
#                         fig, ax = plt.subplots()
#                         plt.pie(a, labels=b, autopct='%1.1f%%')
#                         ax.set_title(f'{param_3} distribution by {param_3}', fontsize=14)
#                         st.pyplot(fig)
#                     else:
#                         st.warning("Please Select Your Variables")
#                 else:
#                     st.warning("Por favor, carga un archivo CSV o XLSX.")
            

        
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