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
import itertools
from scipy import stats

@st.cache_data
def perform_pca(data, target_column, categorical_columns, variance_percentage):
    """
    Performs Principal Component Analysis (PCA) on numerical columns of a DataFrame,
    retaining components that explain a specified percentage of variance.

    Args:
        data (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        categorical_columns (list): A list of categorical column names.
        variance_percentage (float): The desired percentage of variance to explain.

    Returns:
        tuple: A tuple containing:
            - df_final (pd.DataFrame): The DataFrame with reduced dimensions after PCA.
            - total_cols_before (int): The total number of columns before PCA.
            - total_cols_after (int): The total number of columns after PCA.
            - cum_var (np.ndarray): The cumulative explained variance ratios.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Identify numerical columns (excluding the target and categorical columns)
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Standardize the numerical columns
    X[numerical_columns] = StandardScaler().fit_transform(X[numerical_columns])

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X[numerical_columns])

    # Create DataFrame with principal components
    n_components = pca.n_components_
    pc_col_names = [f"PC_{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(data=X_pca, columns=pc_col_names)

    # Calculate the cumulative percentage of explained variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    # Determine the number of components needed to explain variance_percentage% of the variance
    n_components_to_keep = np.argmax(cum_var >= variance_percentage / 100) + 1

    # Keep only the selected principal components
    df_pca_reduced = df_pca.iloc[:, :n_components_to_keep]

    # Add back the target column and categorical columns
    df_final = pd.concat([df_pca_reduced, data[categorical_columns], y], axis=1)

    # Calculate the total number of columns before and after PCA
    total_cols_before = X.shape[1] + len(categorical_columns)
    total_cols_after = df_final.shape[1] - 1  # Exclude the target column

    return df_final, total_cols_before, total_cols_after, cum_var

def plot_cumulative_variance(cum_var, variance_percentage):
    # Plot the cumulative percentage of explained variance for each component
    fig = go.Figure()

    # Line plot
    fig.add_trace(go.Scatter(x=np.arange(1, len(cum_var) + 1), y=cum_var*100, mode='lines', line=dict(color='blue', width=2), name='Explained Variance (%)'))

    # Scatter plot
    fig.add_trace(go.Scatter(x=np.arange(1, len(cum_var) + 1), y=cum_var*100, mode='markers', marker=dict(size=8, color='black'), name='Explained Variance (%)'))

    # Horizontal red line
    fig.add_shape(dict(type="line", x0=1, x1=len(cum_var), y0=variance_percentage, y1=variance_percentage,line=dict(color="red", width=2, dash="dash"),))

    # Layout settings
    fig.update_layout(title="PCA", xaxis_title="Components", yaxis_title="Cוumilative Explained Variance (%)", showlegend=True)

    # Display the plot using Streamlit
    st.plotly_chart(fig)

@st.cache_data
def time_series_feature_extraction(data, target_col, categorical_columns):
    """
    Extract time series features from a DataFrame.

    Parameters
    ----------
    data : pandas DataFrame
        The input DataFrame with hyperspectral data.
    target_col : str
        The name of the target column in the DataFrame.
    categorical_columns : list 
        A list of column names that are categorical and you want to keep in the DataFrame.

    Returns
    -------
    pandas DataFrame
        The output DataFrame with computed features and the target column.
    """
    # Save the target data 
    target_col_data = data[target_col].values

    # Store categorical columns to add back later
    categorical_columns_data = data[categorical_columns]
    
    # exclude non numeric columns and categorical columns
    data = data.drop(columns=categorical_columns)
    data = data.select_dtypes(include=np.number)

    # Get the hyperspectral data columns
    data_cols = [col for col in data.columns if col != target_col]

    data = data[data_cols]

    # calculate means
    means = data.mean(axis=1).values

    # calculate medians
    medians = data.median(axis=1).values

    # calculate standard deviations
    stds = data.std(axis=1).values

    # calculate percent of data beyond 1 std
    percent_beyond_1_std = data.apply(lambda x: np.sum(np.abs(x - x.mean()) > x.std()) / len(x), axis=1)
    percent_beyond_1_std = percent_beyond_1_std.values

    # calculate amplitudes
    amplitudes = data.apply(lambda row: np.ptp(row), axis=1)
    amplitudes = amplitudes.values

    # calculate max values
    maxs = data.max(axis=1).values

    # calculate min values
    mins = data.max(axis=1).values

    # calculate max slopes
    max_slopes = data.apply(lambda row: np.max(np.abs(np.diff(row))), axis=1)
    max_slopes = max_slopes.values

    # calculate median absolute deviations (MAD)
    mads = data.apply(lambda row: np.median(np.abs(row - np.median(row))), axis=1)
    mads = mads.values

    # calculate percent close to median
    percent_close_to_median = data.apply(lambda x: np.sum(np.abs(x - np.median(x)) < 0.5 * np.median(x)) / len(x) * 100, axis=1)
    percent_close_to_median = percent_close_to_median.values

    # calculate skewness
    skewness = data.apply(lambda x: x.skew(), axis=1)
    skewness = skewness.values

    # calculate flux percentile
    flux_percentile = data.quantile(q=0.9, axis=1)

    # calculate percent difference in flux percentile
    percent_difference = flux_percentile.pct_change().fillna(0)
    percent_difference = percent_difference.values

    # define the weights as a numpy array of the column names
    wavelengths = np.array(data.columns)

    # convert the column names to floats if necessary
    if wavelengths.dtype == 'object':
        wavelengths = wavelengths.astype(float)

    # calculate the weighted average for each row in the DataFrame
    weighted_average = data.apply(lambda row: np.average(row, weights=wavelengths), axis=1)
    weighted_average = weighted_average.values

    # create a new DataFrame to store the parameter values for each row
    parameters_df = pd.DataFrame({
        'Mean': means,
        'Median': medians,
        'Std': stds,
        'Percent_Beyond_Std': percent_beyond_1_std,
        'Amplitude': amplitudes,
        'Max': maxs,
        'Min': mins,
        'Max_Slope': max_slopes,
        'MAD': mads,
        'Percent_Close_to_Median': percent_close_to_median,
        'Skew': skewness,
        'Flux_Percentile': flux_percentile.values,
        'Percent_Difference_Flux_Percentile': percent_difference,
        'Weighted_Average': weighted_average
    })

    # Add the categorical columns data back to the DataFrame
    parameters_df = pd.concat([parameters_df, categorical_columns_data], axis=1)
        
    # Add the target column
    parameters_df[target_col] = target_col_data
    
    return parameters_df

@st.cache_data
def NDSI_pearson(data, target_col):
    '''
    Calculates the Pearson correlation coefficient and p-value
    between the normalized difference spectral index (NDSI) and the target column.

    Parameters:
    - data: DataFrame, input data containing spectral bands and target column
    - target_col: str, name of the target column

    Returns:
    - df_results: DataFrame, contains band pairs, Pearson correlation, p-value, and absolute Pearson correlation
    '''

    # Extract labels column
    y = data[target_col].values
    # Delete target column from features dataframe
    df = data.drop(target_col, axis=1)
    # Convert column names to str
    df.columns = df.columns.map(str)
    bands_list = df.columns

    # All possible pairs of columns
    all_pairs = list(itertools.combinations(bands_list, 2))

    # Initialize arrays for correlation values and p-values
    corrs = np.zeros(len(all_pairs))
    pvals = np.zeros(len(all_pairs))

    # Calculate the NDSI and Pearson correlation
    progress_bar = st.progress(0)
    for index, pair in enumerate(all_pairs):
        a = df[pair[0]].values
        b = df[pair[1]].values
        Norm_index = (a - b) / (a + b)
        # Pearson correlation and p-value
        corr, pval = stats.pearsonr(Norm_index, y)
        corrs[index] = corr
        pvals[index] = pval
        # Update progress bar
        progress_bar.progress((index + 1) / len(all_pairs))

    # Convert results to DataFrame
    col1 = [tple[0] for tple in all_pairs]
    col2 = [tple[1] for tple in all_pairs]
    index_col = [f"{tple[0]},{tple[1]}" for tple in all_pairs]
    data = {'band1': col1, "band2": col2, 'Pearson_Corr': corrs, 'p_value': pvals}
    df_results = pd.DataFrame(data=data, index=index_col)
    df_results["Abs_Pearson_Corr"] = df_results["Pearson_Corr"].abs()
    
    return df_results.sort_values('Abs_Pearson_Corr', ascending=False)

@st.cache_data
def display_ndsi_heatmap(results):
    # Pivot the dataframe to have bands as rows and columns
    corr_matrix = results.pivot(index='band1', columns='band2', values='Pearson_Corr')

    # User input for threshold value
    threshold = 0.5
    
    # Set maximum distance for local maxima and minima
    max_distance = 10
    
    data = corr_matrix
    # Find local maxima and minima exceeding the threshold
    local_max = (maximum_filter(data, footprint=np.ones((max_distance, max_distance))) == data) & (data > threshold)
    local_min = (minimum_filter(data, footprint=np.ones((max_distance, max_distance))) == data) & (data < -threshold)

    # Create a Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',  # Choose the color scale
        zmin=-1, zmax=1,  # Set the color scale range
        colorbar=dict(title='Pearson Correlation')  # Add colorbar title
    ))
    
    # Add local maxima and minima
    maxima_x, maxima_y = np.where(local_max)
    fig.add_trace(go.Scatter(x=corr_matrix.columns[maxima_y], y=corr_matrix.index[maxima_x], mode='markers', marker=dict(color='red'), name='Local Maxima'))

    minima_x, minima_y = np.where(local_min)
    fig.add_trace(go.Scatter(x=corr_matrix.columns[minima_y], y=corr_matrix.index[minima_x], mode='markers', marker=dict(color='blue'), name='Local Minima'))

    # Update layout
    fig.update_layout(
        title='NDSI',
        xaxis_title='Band 2',
        yaxis_title='Band 1',
        height=600,  # Adjust height as needed
        width=800,  # Adjust width as needed
        template='plotly'  # Choose plotly theme
    )

    # Display the Plotly figure using st.plotly_chart()
    st.plotly_chart(fig)
    
@st.cache_data
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

