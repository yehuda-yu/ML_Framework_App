import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder


# Helper function to create a download link for a file
def get_binary_file_downloader_html(bin_file, file_label="File"):
    import base64
    bin_str = base64.b64encode(bin_file.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">Download {file_label}</a>'
    return href
     
def perform_eda(data,handle_missing_values,handle_outliers,normalize_data,encode_categorical_variables):
     with st.spinner('Performing EDA...'):
        # Handle missing values
        if handle_missing_values:
            data = data.dropna() # drop rows with missing values
        
        # Handle outliers using the Z-score method
        if handle_outliers:
            z_scores = (data - data.mean()) / data.std()
            data = data[(z_scores < 3).all(axis=1)]
        
        # Normalize the data using min-max scaling
        if normalize_data:
            scaler = MinMaxScaler()
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        
        # Encode categorical variables using one-hot encoding
        if encode_categorical_variables:
            categorical_columns = data.select_dtypes(include=['object']).columns
            encoder = OneHotEncoder()
            encoded_data = encoder.fit_transform(data[categorical_columns])
            data = pd.concat([data.drop(categorical_columns, axis=1), pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out())], axis=1)
        
        # Visualize the relationships between the variables using a pairplot
        #if visualize_data:
         #   fig = plt.figure()
          #  sns.pairplot(data)
           # st.pyplot(fig)
