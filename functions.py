import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
  
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

def train_models(models, param_grids, X_train, y_train):
    best_models = {}  # Store the best models for each type
    best_scores = {}  # Store the best scores for each type
    best_params = {}  # Store the best parameters for each type

    for model_type in models.keys():
        print(f"Training {model_type} model...")
        # Perform the randomized search with cross-validation
        search = RandomizedSearchCV(models[model_type], param_grids[model_type], cv=3, n_iter=10, random_state=42)
        search.fit(X_train, y_train)

        # Store the best model, score, and parameters
        best_models[model_type] = search.best_estimator_
        best_scores[model_type] = round(search.best_score_, 2)
        best_params[model_type] = search.best_params_

    return best_models, best_scores, best_params
def evaluate_models(best_models, X_test, y_test):
    # Create a DataFrame to store the evaluation results
    results = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2", "RPD"])

    # Evaluate each model on the test set and store the results in a dictionary
    model_evaluations = {}
    for model_type, model in best_models.items():
        # Predict the target variable for the test set
        y_test_pred = model.predict(X_test)

        # Calculate MAE, MSE, RMSE, R2, etc.
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_test_pred)
        rpd = y_test.std() / rmse

        # Append the results to the dataframe
        results = pd.concat([results, pd.DataFrame({"Model": [model_type],
                                                    "MAE": [mae],
                                                    "MSE": [mse],
                                                    "RMSE": [rmse],
                                                    "R2": [r2],
                                                    "RPD": [rpd]})])

        # Store the model evaluation results in the dictionary
        model_evaluations[model_type] = {
            "y_test": y_test,
            "y_test_pred": y_test_pred
        }

    return results, model_evaluations

def plot_scatter_subplots(model_evaluations):
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    
    for i, (model_type, evaluation) in enumerate(model_evaluations.items()):
        ax = axes[i]
        ax.scatter(evaluation["y_test_pred"], evaluation["y_test"], alpha=0.8, color='#2a9d8f', edgecolors='black')
        ax.plot([evaluation["y_test"].min(), evaluation["y_test"].max()],
                [evaluation["y_test"].min(), evaluation["y_test"].max()],
                'k--', lw=2)
        ax.set_xlabel('Predictions (y_test_pred)')
        ax.set_ylabel('True Values (y_test)')
        ax.set_title(model_type)

    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
'''
# Plot prediction vs true test set
def plot_scatter_subplots(y_test_dict, y_test_pred_dict, model_names):
    """
    Plots scatter plots of the true values and predictions for each model.
    Args:
        y_test_dict (dict): A dictionary of the true values for each model.
        y_test_pred_dict (dict): A dictionary of the predictions for each model.
        model_names (list): A list of the model names.
    Returns:
        None.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        ax.scatter(y_test_dict[model_name], y_test_pred_dict[model_name], alpha=0.7, edgecolors='w')
        ax.plot([y_test_dict[model_name].min(), y_test_dict[model_name].max()],
                [y_test_dict[model_name].min(), y_test_dict[model_name].max()],
                'k--', lw=2)
        ax.set_xlabel('True Values (y_test)')
        ax.set_ylabel('Predictions (y_test_pred)')
        ax.set_title(model_name)

    plt.tight_layout()
    st.pyplot(fig)
        
        # Visualize the relationships between the variables using a pairplot
        #if visualize_data:
         #   fig = plt.figure()
          #  sns.pairplot(data)
           # st.pyplot(fig)
'''
