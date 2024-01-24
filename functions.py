import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
from sklearn.inspection import PartialDependenceDisplay

st.cache_data
def perform_pca(data, target_column, categorical_columns, variance_percentage):
    # Separate features and target
    X = data.drop([target_column], axis=1)
    y = data[target_column]

    # Identify numerical columns (excluding the target and categorical columns)
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Standardize the numerical columns
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X[numerical_columns])

    # Calculate the cumulative and individual percentage of explained variance for each component
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    individual_var = pca.explained_variance_ratio_

    # Determine the number of components to explain the specified variance_percentage
    n_components = np.argmax(cum_var >= variance_percentage) + 1

    # Get the column names for the selected components
    pc_col_names = [f"PC_{i+1}" for i in range(n_components)]

    # Create a DataFrame with the selected components, categorical columns, and target column
    result_df = pd.DataFrame(data=np.column_stack([X_pca[:, :n_components], X[categorical_columns], y]),
                             columns=pc_col_names + categorical_columns + [target_column])

    # Calculate the total number of columns before and after PCA
    total_cols_before = X.shape[1] + len(categorical_columns) + 1  # +1 for the target column
    total_cols_after = result_df.shape[1]

    return result_df, total_cols_before, total_cols_after, cum_var, individual_var


def plot_cumulative_variance(cum_var):
    # Plot the cumulative percentage of explained variance for each component
    fig = go.Figure()

    # Line plot
    fig.add_trace(go.Scatter(x=np.arange(1, len(cum_var) + 1), y=cum_var*100, mode='lines', line=dict(color='blue', width=2), name='Explained Variance (%)'))

    # Scatter plot
    fig.add_trace(go.Scatter(x=np.arange(1, len(cum_var) + 1), y=cum_var*100, mode='markers', marker=dict(size=8, color='black'), name='Explained Variance (%)'))

    # Layout settings
    fig.update_layout(title="PCA",
                      xaxis_title="Components",
                      yaxis_title="Explained Variance (%)",
                      showlegend=True,
                      height=700,
                      width=1000)

    # Display the plot using Streamlit
    st.plotly_chart(fig)


st.cache_data
def replace_missing_with_average(data):
    """Replace missing values with the average of each column."""
    return data.fillna(data.mean())

st.cache_data
def replace_missing_with_zero(data):
    """Replace missing values with zero."""
    return data.fillna(0)

st.cache_data
def delete_missing_values(data):
    """Delete rows containing missing values."""
    return data.dropna()

st.cache_data
def normalize_data_minmax(data):
    """Normalize data using Min-Max scaling."""
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, columns=data.columns)

st.cache_data
def normalize_data_standard(data):
    """Standardize data using Z-score standardization."""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return pd.DataFrame(standardized_data, columns=data.columns)

st.cache_data
def encode_categorical_onehot(data):
    """One-hot encode categorical variables."""
    categorical_columns = data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
    data = pd.concat([data, encoded_data], axis=1)
    data = data.drop(categorical_columns, axis=1)
    return data

st.cache_data
def encode_categorical_label(data):
    """Label encode categorical variables."""
    le = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = le.fit_transform(data[column])
    return data

st.cache_data
def train_models(models, param_grids, X_train, y_train):
    try:
        best_models = {}  # Store the best models for each type
        best_scores = {}  # Store the best scores for each type
        best_params = {}  # Store the best parameters for each type

        for model_type in models.keys():
            st.write(f"Training {model_type} model...")
            # Perform the randomized search with cross-validation
            search = RandomizedSearchCV(models[model_type], param_grids[model_type], cv=3, n_iter=10, random_state=42)
            search.fit(X_train, y_train)

            # Store the best model, score, and parameters
            best_models[model_type] = search.best_estimator_
            best_scores[model_type] = round(search.best_score_, 2)
            best_params[model_type] = search.best_params_

        return best_models, best_scores, best_params

    except Exception as e:
        st.error(f"An error occurred while training models: {e}")
        return None, None, None

st.cache_data
def evaluate_models(best_models, X_test, y_test):
    try:
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

    except Exception as e:
        st.error(f"An error occurred while evaluating models: {e}")
        return None, None

st.cache_data
def plot_scatter_subplots(model_evaluations):
    try:
        fig = sp.make_subplots(rows=1, cols=len(model_evaluations),
                               subplot_titles=list(model_evaluations.keys()))

        colors = ['#2a9d8f', '#e76f51', '#f4a261', '#e9c46a', '#264653']  # Example color palette
        color_iter = iter(colors)

        for i, (model_type, evaluation) in enumerate(model_evaluations.items()):
            color = next(color_iter)  # Assign a distinct color to each model

            scatter_trace = go.Scatter(x=evaluation["y_test_pred"],
                                       y=evaluation["y_test"],
                                       mode='markers',
                                       marker=dict(color=color, line=dict(color='black', width=1)),
                                       name=model_type)

            reference_line = go.Scatter(x=[min(evaluation["y_test"]), max(evaluation["y_test"])],
                                        y=[min(evaluation["y_test"]), max(evaluation["y_test"])],
                                        mode='lines',
                                        line=dict(color='black', dash='dash'),
                                        showlegend=False)

            fig.add_trace(scatter_trace, row=1, col=i+1)
            fig.add_trace(reference_line, row=1, col=i+1)

            fig.update_xaxes(title_text="Predictions (test set)", row=1, col=i+1)
            fig.update_yaxes(title_text="True Values (test set)", row=1, col=i+1)

        fig.update_layout(title_text="Scatter Subplots",
                          margin=dict(l=0, r=0, t=60, b=0))

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting scatter subplots: {e}")
        

st.cache_data       
def plot_feature_importance(best_models, X_train, y_train):
    try:
        colors = ['#2a9d8f', '#e76f51', '#f4a261']   # Color palette for 3 models
        color_iter = iter(colors)

        fig = sp.make_subplots(rows=1, cols=len(best_models),
                               specs=[[{'type': 'pie'}] * len(best_models)],
                               subplot_titles=list(best_models.keys()))

        for i, (model_type, model) in enumerate(best_models.items()):
            color = next(color_iter)  # Assign next color from palette

            if hasattr(model, 'feature_importances_'):  # For Random Forest
                importances = model.feature_importances_
            else:  # For SVM Regression and other models
                result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
                importances = result.importances_mean

            indices = np.argsort(importances)[::-1]
            names = [X_train.columns[i] for i in indices]
            importance_values = [importances[i] for i in indices]

            fig.add_trace(go.Pie(labels=names, values=importance_values,
                                  textinfo='label+percent', hole=0.3,
                                  marker=dict(colors=[color] * len(importances),line=dict(color='white', width=2)),  # Set color for pie chart
                                  title=model_type),
                          row=1, col=i+1)

        fig.update_layout(title_text="Feature Importance",
                          margin=dict(l=0, r=0, t=60, b=0))

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting feature importance: {e}")

st.cache_data
def plot_pdp(best_models, X_train, features, target_column):
    try:
        colors = ['#2a9d8f', '#e76f51', '#f4a261', '#738bd7', '#d35400', '#a6c7d8']

        for selected_feature in features:
            st.subheader(f"Partial Dependence Plots (PDP) for {selected_feature}")
            fig, axs = plt.subplots(1, len(best_models), figsize=(15, 6), constrained_layout=True)

            for i, (model_name, model) in enumerate(best_models.items()):
                features_info = {
                    "features": [selected_feature],
                    "kind": "average",
                }

                display = PartialDependenceDisplay.from_estimator(
                    model,
                    X_train,
                    **features_info,
                    ax=axs[i],
                )

                axs[i].set_facecolor(colors[i % len(colors)])  # Cycle through colors
                axs[i].set_title(f"{model_name}")
                axs[i].set_xlabel(selected_feature)
                axs[i].set_ylabel(f"Partial Dependence for {target_column}")

            fig.suptitle(f"Partial Dependence of {target_column} on {selected_feature}")
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting PDP: {e}")

