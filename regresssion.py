import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import Linear
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from torch.utils.data import TensorDataset

def main():
    data = st.file_uploader("📁 Load Data File", type=["csv", "xlsx"])
    if data is not None:
        try:
            if data.name.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.name.endswith('.xlsx'):
                df = pd.read_excel(data)
        except Exception as e:
                st.error(f"Ocurrió un error al leer el archivo: {e}")
        st.write(df)
        columns = list(df.columns)
        viz = ['bar 📊', 'pie 🥧', 'correlation 🎢', 'Line 🪡','Classification 🌍', 'Trend 📈']
        tabs = st.tabs(viz)
        date_columns = [col for col in columns if 'year' in col.lower() or 'date' in col.lower()]

        # if len(date_columns) == 0:
        #     st.warning("No Date or Year column found")
        #     # return

        with tabs[0]:
            if len(date_columns) == 0:
                st.warning("No Date or Year column found")
            else:
                date_field = st.selectbox("Select Date Field for Bar Plot", date_columns)
                if date_field:
                    if 'year' in date_field.lower():
                        df[date_field] = pd.to_numeric(df[date_field], errors='coerce')  # Asegurar que sea numérico
                        df = df.dropna(subset=[date_field])  # Eliminar valores NaN
                        min_year = int(df[date_field].min())
                        max_year = int(df[date_field].max())

                        year_range = st.slider('Select Year range for Bar Plot',
                                            min_value=min_year,
                                            max_value=max_year,
                                            value=(min_year, max_year))

                        df_filtered = df[(df[date_field] >= year_range[0]) & (df[date_field] <= year_range[1])]

                    elif 'date' in date_field.lower():
                        df[date_field] = pd.to_datetime(df[date_field], errors='coerce')
                        df = df.dropna(subset=[date_field])
                        min_date = df[date_field].min().date()
                        max_date = df[date_field].max().date()

                        year_range = st.slider('Select Date range for Bar Plot',
                                            min_value=min_date,
                                            max_value=max_date,
                                            value=(min_date, max_date))

                        df_filtered = df[(df[date_field].dt.date >= year_range[0]) & (df[date_field].dt.date <= year_range[1])]
                        

                    numeric_columns = df_filtered.select_dtypes(include='number').columns.tolist()
                    string_columns = df_filtered.select_dtypes(include='object').columns.tolist()

                    param_1 = st.selectbox("Select X value for Barplot", string_columns, placeholder="Select one parameter",index=None)
                    param_2 = st.selectbox("Select Y value for Barplot", numeric_columns, placeholder="Select one parameter",index=None)
                    filter = st.selectbox("Select filter for Barplot", string_columns, placeholder="Select one parameter",index=None)

                    if param_1 and param_2 or filter:
                        fig, ax = plt.subplots()
                        sns.barplot(data=df_filtered, x=param_1, y=param_2, hue=filter, ax=ax, errorbar=None)
                        ax.set_title(f'Total {param_2} by {param_1} filtering {filter}', fontsize=14)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                        st.pyplot(fig)
                    else:
                        st.warning("Please Select Your Variables")
                else:
                    st.warning("Por favor, selecciona un campo de fecha válido.")

        with tabs[1]:
            if len(date_columns) == 0:
                st.warning("No Date or Year column found")
            else:
                date_field_pie = st.selectbox("Select Date Field for Pie Chart", date_columns)
                if date_field_pie:
                    if 'year' in date_field_pie.lower():
                        df[date_field_pie] = pd.to_numeric(df[date_field_pie], errors='coerce')
                        df = df.dropna(subset=[date_field_pie])
                        min_year = int(df[date_field_pie].min())
                        max_year = int(df[date_field_pie].max())

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

                        year_range_2 = st.slider('Select Date range for Pie Chart',
                                                min_value=min_date,
                                                max_value=max_date,
                                                value=(min_date, max_date))

                        df_filtered = df[(df[date_field_pie].dt.date >= year_range_2[0]) & (df[date_field_pie].dt.date <= year_range_2[1])]

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

        with tabs[3]:
            if len(date_columns) == 0:
                st.warning("No Date or Year column found")
            else:
                numeric_columns_4 = df_filtered.select_dtypes(include='number').columns.tolist()
                if numeric_columns_4:
                    x_columns = st.multiselect("Select X values (multiple numeric columns)",
                                                numeric_columns_4,
                                                default=None)
                    y_column = st.selectbox("Select Y value (Target)", [""] + numeric_columns_4, index=0)  
                    # st.write(type(x_columns))

                    if x_columns and y_column:
                        st.write("Selected X columns:", ", ".join(x_columns))
                        params_selected = x_columns
                        # st.write(params_selected)
                        st.write("Selected Y column:", y_column)
                        X_series = df_filtered[x_columns] 
                        y_series = df_filtered[y_column]
                        
                        fig, ax = plt.subplots()
                        if len(x_columns)==1:
                            # X = x_columns
                            # y = y_column
                            X = np.array(X_series, dtype=np.float32).reshape(-1, 1)
                            y = np.array(y_series, dtype=np.float32).reshape(-1, 1)
                            scaler_X = StandardScaler()
                            scaler_y = StandardScaler()
                            X_scaled = scaler_X.fit_transform(X)
                            y_scaled = scaler_y.fit_transform(y)
                            X_tensor = torch.from_numpy(X_scaled)
                            y_tensor = torch.from_numpy(y_scaled)
                            class LR(nn.Module):
                                def __init__(self, input_dim, output_dim):
                                    super(LR, self).__init__()
                                    self.linear = nn.Linear(input_dim, output_dim)
                                def forward(self, x):
                                    out = self.linear(x)
                                    return out
                            input_dim = 1
                            output_dim = 1
                            learning_rate = 0.01
                            epochs = 100 
                            model = LR(input_dim, output_dim)      
                            criterion = nn.MSELoss()
                            optimizer = optim.SGD(model.parameters(), lr=learning_rate)  
                            for epoch in range(epochs):
                                model.train() 
                                y_pred = model(X_tensor)
                                loss = criterion(y_pred, y_tensor)
                                optimizer.zero_grad() 
                                loss.backward()  
                                optimizer.step() 
                                if (epoch + 1) % 100 == 0:
                                    st.write(f'Training Complete, model Loss: {loss.item():.4f}')
                                    
                            weights = model.linear.weight.data.numpy()
                            bias = model.linear.bias.data.numpy()
                            st.divider()
                            st.write(f'Weights: {weights.flatten()[0]:.4f}')
                            st.write(f'Bias: {bias.flatten()[0]:.4f}')
                            model.eval()  
                            with torch.no_grad():
                                predicted = model(X_tensor)  
                                
                                
                        elif len(x_columns)>1:
                            X = np.array(X_series, dtype=np.float32)
                            y = np.array(y_series, dtype=np.float32).reshape(-1, 1)
                            scaler_X = StandardScaler()
                            scaler_y = StandardScaler()
                            X_scaled = scaler_X.fit_transform(X)
                            y_scaled = scaler_y.fit_transform(y)
                            X_tensor = torch.from_numpy(X_scaled)
                            y_tensor = torch.from_numpy(y_scaled)

                            class LR(nn.Module):
                                def __init__(self, input_dim, output_dim):
                                    super(LR, self).__init__()
                                    self.linear = nn.Linear(input_dim, output_dim)
                                def forward(self, x):
                                    out = self.linear(x)
                                    return out
                            input_dim = len(params_selected)
                            output_dim = 1
                            learning_rate = 0.01
                            epochs = 100 
                            model = LR(input_dim, output_dim)
                            criterion = nn.MSELoss()
                            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
                            for epoch in range(epochs):
                                model.train() 
                                y_pred = model(X_tensor)
                                loss = criterion(y_pred, y_tensor)
                                optimizer.zero_grad() 
                                loss.backward()  
                                optimizer.step() 
                                if (epoch + 1) % 100 == 0:
                                    st.write(f'Training Complete, model Loss: {loss.item():.4f}')
                                    
                            weights = model.linear.weight.data.numpy()
                            bias = model.linear.bias.data.numpy()
                            st.divider()
                            for i, param in enumerate(params_selected):
                                st.write(f'Weight for {param}: {weights.flatten()[i]:.4f}')
                            st.write(f'Bias: {bias.flatten()[0]:.4f}')
                
                            with torch.no_grad():
                                predicted = model(X_tensor)
        
        with tabs[4]:
            columns = list(df.columns)
            # numeric_columns_5 = df.select_dtypes().columns.tolist()
        
            x_columns = st.multiselect("Select X values (multiple numeric columns)",
                                                columns,
                                                default=None)
            y_column = st.selectbox("Select Y value (Target)", columns[-1])
            
            if df[y_column].dtype == 'object':
                encoder = LabelEncoder()
                df["y_column_2"] = encoder.fit_transform(df[y_column]) 
                X= df.drop([y_column,'y_column_2'],axis=1)
                Y = df['y_column_2']
            else:
                X= df.drop([y_column],axis=1)
                Y = df[y_column]
              
            X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

            X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            input_dim = X.shape[1] 
            output_dim = len(Y.unique())
            hiden_layers = 10
            Soft_Max = torch.nn.Sequential( torch.nn.Linear(input_dim,hiden_layers),
                                            torch.nn.Sigmoid(),
                                            torch.nn.Linear(hiden_layers,output_dim),
                                            torch.nn.Softmax(dim=1))

            def train_model(model,n_epochs,learning_rate,train_batch_size,validation_bath_size,train_dataset,validation_dataset):
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size,shuffle=True)
                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=validation_bath_size)
                loss_list = []
                accuracy_list = []
                for epoch in range(n_epochs):
                    for x, y in train_loader:
                        optimizer.zero_grad()
                        z = model(x)
                        loss = criterion(z, y)
                        loss.backward()
                        optimizer.step()
                    model.eval()  
                    correct = 0
                    N_test = len(validation_dataset)  
                    with torch.no_grad():
                        for x_test, y_test in validation_loader:
                            z = model(x_test.view(-1,input_dim))
                            _, yhat = torch.max(z.data, 1)
                            correct += (yhat == y_test).sum().item()
                    accuracy = correct / N_test
                    loss_list.append(loss.data)
                    accuracy_list.append(accuracy)
                return loss_list, accuracy_list

            n_epochs = 100
            learning_rate = 0.1
            train_batch_size = 100
            validation_bath_size = 10
            train_dataset = train_dataset
            validation_dataset = val_dataset
            N_test = len(validation_dataset)
            loss_list, accuracy_list = train_model(Soft_Max,n_epochs,learning_rate,train_batch_size,validation_bath_size,train_dataset,validation_dataset)
            
            predicho = st.text_input("Write the values to predict a class in the same order whic shows in de DataFrame, use comma to separate the values:")
            
            try: 
                valores = [float(valor.strip()) for valor in predicho.split(",")]
                x =  torch.tensor(valores,dtype=torch.float32)
                z = Soft_Max(x.reshape(-1,input_dim))
                _, yhat = torch.max(z, 1)
                if df[y_column].dtype == 'object':
                    a = (df[y_column].unique())
                    b = (df['y_column_2'].unique())
                    c = dict(zip(b,a))
                else:
                    c = (df[y_column].unique())   
                for i in range(len(c)):
                    if int(yhat[0]) in c:
                        st.write(f'Class predicted with the values enteres is: {c[i]}')
                        st.write(f'clas probability: {_.item()}')
                        break
            except ValueError:
                st.write("Error: please use numeric values separated by commas.")


                
                
if __name__ == "__main__":
    main()