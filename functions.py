import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import itertools
from scipy import stats
from scipy.ndimage.filters import maximum_filter, minimum_filter
from sklearn.linear_model import LassoCV, LassoLarsCV, LarsCV, Lasso, OrthogonalMatchingPursuitCV, LassoLars, OrthogonalMatchingPursuit, ElasticNetCV, ElasticNet, TweedieRegressor, HuberRegressor, RANSACRegressor, LinearRegression, BayesianRidge, Ridge, TransformedTargetRegressor, LassoLarsIC, Lars, PassiveAggressiveRegressor, SVR, NuSVR, SGDRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import uniform, randint
from scipy.stats import loguniform
from lazypredict.Supervised import LazyRegressor

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
def NDSI_pearson(data,categorical_columns,  target_col):
    '''
    Calculates the Pearson correlation coefficient and p-value
    between the normalized difference spectral index (NDSI) and the target column.

    Parameters:
    - data: DataFrame, input data containing spectral bands and target column
    - categorical_columns: list, list of the categorial colums to drop before ndsi
    - target_col: str, name of the target column

    Returns:
    - df_results: DataFrame, contains band pairs, Pearson correlation, p-value, and absolute Pearson correlation
    '''

    # Extract labels column
    y = data[target_col].values
    # Delete target column from features dataframe
    df = data.drop(target_col, axis=1)
    # drop non numeric columns
    df = data.drop(categorical_columns, axis=1)
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
def display_ndsi_heatmap(results, threshold, max_distance):
    """
    Display a heatmap with local minima and maxima points based on Pearson correlation values.

    Parameters:
    - results : DataFrame
        DataFrame containing Pearson correlation values between spectral bands.
    - threshold : float
        Threshold value for identifying local minima and maxima points.
    - max_distance : int
        Maximum distance for local minima and maxima identification.

    Returns:
    - top_bands_list : list of tuples
        List of tuples where each tuple contains the names of two spectral bands
        corresponding to the local minima and maxima points.
    """
    # Pivot the dataframe to have bands as rows and columns
    data = results.pivot(index='band1', columns='band2', values='Pearson_Corr')

    # Find local maxima and minima exceeding the threshold
    local_max = (maximum_filter(data, footprint=np.ones((max_distance, max_distance))) == data) & (data > threshold)
    local_min = (minimum_filter(data, footprint=np.ones((max_distance, max_distance))) == data) & (data < -threshold)

    # Get indices of local maxima and minima
    maxima_x, maxima_y = np.where(local_max)
    minima_x, minima_y = np.where(local_min)

    # Create lists to store band1 and band2 indices for minima and maxima
    minima_list = [(data.index[minima_x][i], data.columns[minima_y][i]) for i in range(len(minima_x))]
    maxima_list = [(data.index[maxima_x][i], data.columns[maxima_y][i]) for i in range(len(maxima_x))]

    # Merge the two lists into one list
    top_bands_list = minima_list + maxima_list

    # Create a Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='RdBu_r',  # Choose the color scale
        zmin=-1, zmax=1,  # Set the color scale range
        colorbar=dict(title='Pearson Correlation')  # Add colorbar title
    ))
    
    # Add local maxima and minima
    maxima_x, maxima_y = np.where(local_max)
    fig.add_trace(go.Scatter(x=data.columns[maxima_y], y=data.index[maxima_x], mode='markers', marker=dict(color='red'), name='Local Maxima'))

    minima_x, minima_y = np.where(local_min)
    fig.add_trace(go.Scatter(x=data.columns[minima_y], y=data.index[minima_x], mode='markers', marker=dict(color='blue'), name='Local Minima'))

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

    return top_bands_list

@st.cache_data
def calculate_ndsi(data, top_bands_list):
    """
    Calculate Normalized Difference Spectral Index (NDSI) for each pair of spectral bands.

    Parameters:
    - data : DataFrame
        Original data containing spectral bands.
    - top_bands_list : list of tuples
        List where each tuple contains the names of two spectral bands.

    Returns:
    - ndsi_df : DataFrame
        DataFrame where each column represents a pair of spectral bands,
        and the values are the corresponding NDSI values calculated as (a - b) / (a + b),
        where 'a' and 'b' are the values of the respective spectral bands.
    """
    ndsi_df = pd.DataFrame()

    # Calculate NDSI for each tuple in the list
    for tup in top_bands_list:
        a = data[tup[0]]
        b = data[tup[1]]
        ndsi = (a - b) / (a + b)
        column_name = f"{tup[0]}-{tup[1]}"
        ndsi_df[column_name] = ndsi

    return ndsi_df

@st.cache_data
def replace_missing_with_average(data):
    """Replace missing values with the average of each column."""
    return data.fillna(data.mean())

@st.cache_data
def replace_missing_with_zero(data):
    """Replace missing values with zero."""
    return data.fillna(0)

@st.cache_data
def delete_missing_values(data):
    """Delete rows containing missing values."""
    return data.dropna()

@st.cache_data
def normalize_data_minmax(data):
    """Normalize data using Min-Max scaling."""
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return pd.DataFrame(normalized_data, columns=data.columns)

@st.cache_data
def normalize_data_standard(data):
    """Standardize data using Z-score standardization."""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return pd.DataFrame(standardized_data, columns=data.columns)

@st.cache_data
def encode_categorical_onehot(data):
    """One-hot encode categorical variables."""
    categorical_columns = data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
    data = pd.concat([data, encoded_data], axis=1)
    data = data.drop(categorical_columns, axis=1)
    return data

@st.cache_data
def encode_categorical_label(data):
    """Label encode categorical variables."""
    le = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = le.fit_transform(data[column])
    return data

@st.cache_data
def evaluate_regression_models(X_train, X_test, y_train, y_test):
    """
    This function evaluates various regression models using LazyRegressor.
    
    Inputs:
    - X_train: Training features (array-like, shape (n_samples, n_features))
    - X_test: Testing features (array-like, shape (n_samples, n_features))
    - y_train: Training target (array-like, shape (n_samples,))
    - y_test: Testing target (array-like, shape (n_samples,))
    
    Returns:
    - models_df: DataFrame containing information about various regression models
    """
    try:
        # Initialize LazyRegressor
        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        
        # Fit LazyRegressor on the data
        models = reg.fit(X_train, y_train)
        
        # Predict with the fitted models
        y_pred = reg.predict(X_test)
        
        # Convert model dictionary to DataFrame
        models_df = pd.DataFrame(models)
        
        return models_df

    except Exception as e:
        st.error(f"An error occurred while training models: {e}")
        return None
@st.cache_data

def tune_LassoCV_model(X_train, y_train):
    try:
        # Define the model
        lasso_cv = LassoCV()

        # Define hyperparameters to tune
        param_dist = {
            "eps": [0.001, 0.01, 0.1],
            # wide range of alpha values
            "n_alphas": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "fit_intercept": [True, False],
            "precompute": ['auto', True, False],
            "max_iter": [500, 1000, 2000, 3000],
            "tol": [0.0001, 0.001, 0.01],
            "cv": [3, 5, 10],  # Cross-validation strategy
            "positive": [False, True],
            "selection": ['cyclic', 'random']
        }
        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_,random_search.best_score_
    
    except Exception as e:
        st.error(f"An error occurred while tuning LassoCV model: {e}")
        return None

@st.cache
def tune_LassoLarsCV_model(X_train, y_train):
    try:
        # Define the model
        lasso_lars_cv = LassoLarsCV()

        # Define hyperparameters to tune
        param_dist = {
            "fit_intercept": [True, False],
            "verbose": [True, False],
            "max_iter": [500, 1000, 1500],
            "precompute": [True, False, 'auto'],
            "max_n_alphas": [500, 1000, 1500],
            "eps": [1e-16, 1e-12, 1e-8],
            "copy_X": [True, False],
            "positive": [True, False]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso_lars_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_score_

    except Exception as e:
        st.error(f"An error occurred while tuning LassoLarsCV model: {e}")
        return None, None

@st.cache
def tune_LarsCV(X_train, y_train):
    try:
        # Define the model
        lars_cv = LarsCV()

        param_dist = {
            'fit_intercept': [True, False],
            'verbose': [False, True],
            'max_iter': randint(100, 1000),
            'precompute': ['auto', True, False],
            'cv': randint(3, 10),
            'max_n_alphas': randint(100, 2000),
            'n_jobs': [-1, None],
            'eps': uniform(1e-8, 1e-3),
            'copy_X': [True, False]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lars_cv, param_distributions=param_dist, n_iter=100, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning LarsCV model: {e}")
        return None, None
        
@st.cache
def tune_OrthogonalMatchingPursuitCV(X_train, y_train):
    try:
        # Define the model
        omp_cv = OrthogonalMatchingPursuitCV()

        param_dist = {
            "copy": [True, False],
            "fit_intercept": [True, False],
            "cv": [3, 5, 10],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(omp_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning OrthogonalMatchingPursuitCV model: {e}")
        return None, None
        
@st.cache
def tune_NuSVR(X_train, y_train):
    try:
        # Define the model
        nusvr = NuSVR()

        param_dist = {
            "nu": uniform(0.1, 0.9),
            "C": uniform(0.1, 10),
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "degree": randint(1, 10),
            "gamma": ['scale', 'auto', uniform(0.001, 1)],
            "coef0": uniform(-1, 1),
            "shrinking": [True, False],
            "tol": uniform(1e-4, 1e-2),
            "cache_size": [100, 200, 300],
            "verbose": [True],
            "max_iter": [-1, 1000, 2000]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(nusvr, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning NuSVR model: {e}")
        return None, None

@st.cache
def tune_lasso(X_train, y_train):
    try:
        # Define the model
        lasso = Lasso()

        # Define hyperparameters to tune
        param_dist = {
            "alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "precompute": ['auto', True, False],
            "copy_X": [True, False],
            "max_iter": [1000, 2000, 3000],
            "tol": [0.0001, 0.001, 0.01],
            "warm_start": [True, False],
            "positive": [False, True],
            "selection": ['cyclic', 'random']
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning Lasso model: {e}")
        return None, None
        
@st.cache
def tune_LassoLarsCV(X_train, y_train):
    try:
        # Define the model
        lasso_lars_cv = LassoLarsCV()

        # Define hyperparameters to tune
        param_dist = {
            "fit_intercept": [True, False],
            "verbose": [True, False],
            "normalize": [True, False],
            "precompute": [True, False, 'auto'],
            "max_iter": [500, 1000, 1500],
            "eps": [1e-16, 1e-12, 1e-8],
            "copy_X": [True, False],
            "positive": [True, False]
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso_lars_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning LassoLarsCV model: {e}")
        return None, None
    
@st.cache
def omp_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        omp = OrthogonalMatchingPursuit()

        # Define hyperparameters to tune
        param_dist = {
                "n_nonzero_coefs": randint(1, X_train.shape[1] // 2),  # Set a reasonable range for n_nonzero_coefs
                "tol": uniform(1e-5, 1e-2),  # Set a reasonable range for tol
                "fit_intercept": [True, False],
                "precompute": ['auto', True, False],
            }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(omp, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning OrthogonalMatchingPursuit model: {e}")
        return None, None
        
@st.cache
def elastic_net_cv_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        elastic_net_cv = ElasticNetCV()

        # Define hyperparameters to tune
        param_dist = {
            "l1_ratio": uniform(0, 0.99),  # Set a reasonable range for l1_ratio
            "eps": [1e-3, 1e-2, 1e-1],  # Set a reasonable range for eps
            "n_alphas": [100, 200, 300, 400, 500],  # Set a reasonable range for n_alphas
            "fit_intercept": [True, False],
            "precompute": ['auto', True, False],
            "max_iter": [500, 1000, 2000, 3000],  # Set a reasonable range for max_iter
            "tol": [1e-4, 1e-3, 1e-2],  # Set a reasonable range for tol
            "cv": [3, 5, 10],  # Set a reasonable range for cv
            "positive": [False, True],
            "selection": ['cyclic', 'random']
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(elastic_net_cv, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning ElasticNetCV model: {e}")
        return None, None
    
@st.cache
def elastic_net_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        elastic_net = ElasticNet()

        # Define hyperparameters to tune
        param_dist = {
            "alpha": uniform(1e-6, 1.0),  # Set a reasonable range for alpha
            "l1_ratio": uniform(0.01, 1.0),  # Set a reasonable range for l1_ratio
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "precompute": ['auto', True, False],
            "max_iter": [1000, 2000, 3000],  # Set a reasonable range for max_iter
            "tol": [1e-4, 1e-3, 1e-2],  # Set a reasonable range for tol
            "warm_start": [True, False],
            "positive": [False, True],
            "selection": ['cyclic', 'random']
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(elastic_net, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning ElasticNet model: {e}")
        return None, None

@st.cache
def tweedie_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        tweedie_regressor = TweedieRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "power": [0, 1, 2],  # Set a reasonable range for power
            "alpha": uniform(0, 1),  # Set a reasonable range for alpha
            "fit_intercept": [True, False],
            "link": ['auto', 'identity', 'log'],
            #"solver": ["lbfgs", "newton-cholesky"],
            "warm_start": [True, False],
            "max_iter": [100, 500, 1000],  # Set a reasonable range for max_iter
            "tol": uniform(1e-5, 1e-2),  # Set a reasonable range for tol
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(tweedie_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning TweedieRegressor model: {e}")
        return None, None
        
@st.cache
def dummy_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        dummy_regressor = DummyRegressor()

        # if y_train is numpy array
        if isinstance(y_train, np.ndarray):
            train_max, train_min, train_mean, train_median = np.max(y_train), np.min(y_train), np.mean(y_train), np.median(y_train)

        elif isinstance(y_train, pd.Series):
            train_max, train_min, train_mean, train_median = y_train.max(), y_train.min(), y_train.mean(), y_train.median()
        
        # Define hyperparameters to tune
        param_dist = {
            "strategy": ["mean", "median", "quantile", "constant"],
            "constant": [0.0, 1.0, -1.0, train_max, train_min, train_mean, train_median],
            "quantile": [0.0, 0.25, 0.5, 0.75, 1.0],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(dummy_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning DummyRegressor model: {e}")
        return None, None
        
@st.cache
def huber_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        huber_regressor = HuberRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "epsilon": uniform(1.0, 2.0),  # Epsilon controls the number of outliers
            "alpha": uniform(1e-6, 1.0),  # Strength of L2 regularization
            "max_iter": [100, 500, 1000],  # Maximum number of iterations
            "fit_intercept": [True, False],
            "tol": uniform(1e-5, 1e-2),  # Tolerance for stopping criterion
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(huber_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning HuberRegressor model: {e}")
        return None, None
        
@st.cache
def svr_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        svr = NuSVR()

        # Define hyperparameters to tune
        param_dist = {
            "nu": uniform(0.1, 0.9),  # An upper bound on the fraction of margin errors and a lower bound of support vectors
            "C": uniform(0.1, 100),  # Regularization parameter
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],  # Type of kernel
            "degree": randint(1, 10),  # Degree of the polynomial kernel
            "gamma": ['scale', 'auto', uniform(0.001, 1)],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            "coef0": uniform(-1, 1),  # Independent term in kernel function
            "shrinking": [True, False],  # Whether to use the shrinking heuristic
            "tol": uniform(1e-4, 1e-2),  # Tolerance for stopping criterion
            "cache_size": [100, 200, 300],  # Size of the kernel cache
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(svr, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning SVR model: {e}")
        return None, None
    
@st.cache
def ransac_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        ransac_regressor = RANSACRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "min_samples": uniform(0.1, 0.5),  # Minimum number of samples for consensus set
            "residual_threshold": uniform(1.0, 5.0),  # Maximum residual for inliers
            "max_trials": [100, 200, 300],  # Maximum number of iterations for random sample selection
            "max_skips": [10, 20, 30],  # Maximum number of iterations that can be skipped
            "stop_n_inliers": [int(0.8 * X_train.shape[0]), int(0.9 * X_train.shape[0])],  # Stop iteration if at least this number of inliers are found
            "stop_score": [0.95, 0.96, 0.97],  # Stop iteration if score is greater equal than this threshold
            "stop_probability": [0.98, 0.99],  # Probability for stopping iteration
            "loss": ["absolute_loss", "squared_loss"],  # Loss function for determining inliers/outliers
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(ransac_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning RANSACRegressor model: {e}")
        return None, None
        
@st.cache
def bayesian_ridge_hyperparam_search(X_train, y_train):
    try:

        # Define the model
        bayesian_ridge = BayesianRidge()

        # Define hyperparameters to tune
        param_dist = {
            "tol": uniform(1e-5, 1e-2),  # Tolerance for stopping criterion
            "alpha_1": uniform(1e-8, 1e-4),  # Shape parameter for Gamma prior over alpha
            "alpha_2": uniform(1e-8, 1e-4),  # Inverse scale parameter for Gamma prior over alpha
            "lambda_1": uniform(1e-8, 1e-4),  # Shape parameter for Gamma prior over lambda
            "lambda_2": uniform(1e-8, 1e-4),  # Inverse scale parameter for Gamma prior over lambda
            "alpha_init": [None] + list(uniform(0.1, 1.0).rvs(10)),  # Initial value for alpha
            "lambda_init": [None] + list(uniform(0.1, 1.0).rvs(10)),  # Initial value for lambda
            "compute_score": [True, False],
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "verbose": [False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(bayesian_ridge, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning BayesianRidge model: {e}")
        return None, None

@st.cache
def ridge_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        ridge = Ridge()

        # Define hyperparameters to tune
        param_dist = {
            "alpha": uniform(0.1, 10),  # Regularization strength
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "max_iter": [100, 500, 1000],  # Maximum number of iterations
            "tol": uniform(1e-5, 1e-2),  # Tolerance for stopping criterion
            "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(ridge, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning Ridge model: {e}")
        return None, None
        
@st.cache
def linear_regression_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        linear_regression = LinearRegression()

        # Define hyperparameters to tune
        param_dist = {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "copy_X": [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(linear_regression, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning LinearRegression model: {e}")
        return None, None
        
@st.cache
def transformed_target_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        transformed_target_regressor = TransformedTargetRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "regressor": [LinearRegression(), Ridge(), BayesianRidge()],
            "transformer": [None, "quantile", "yeo-johnson", "box-cox"],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(transformed_target_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning TransformedTargetRegressor model: {e}")
        return None, None
        
@st.cache
def lasso_lars_ic_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        lasso_lars_ic = LassoLarsIC()

        # Define hyperparameters to tune
        param_dist = {
            "criterion": ["aic", "bic"],
            "normalize": [True, False],
            "fit_intercept": [True, False],
            "max_iter": [500, 1000, 1500],
            "eps": [1e-16, 1e-12, 1e-8],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lasso_lars_ic, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        # Return the best model
        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning LassoLarsIC model: {e}")
        return None, None
        
@st.cache
def lars_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        lars = Lars()

        # Define hyperparameters to tune
        param_dist = {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "precompute": [True, False, 'auto'],
            "n_nonzero_coefs": [100, 200, 300, 400, 500],
            "eps": uniform(1e-16, 1e-8),
            "copy_X": [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lars, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning Lars model: {e}")
        return None, None
        
@st.cache
def mlp_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        mlp_regressor = MLPRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "hidden_layer_sizes": [(100,), (200,), (300,), (400,), (500,)],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["lbfgs", "sgd", "adam"],
            "alpha": loguniform(1e-6, 1.0),
            "batch_size": [16, 32, 64, 128],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": loguniform(1e-4, 1e-2),
            "power_t": loguniform(0.1, 1.0),
            "max_iter": [200, 400, 600, 800, 1000],
            "shuffle": [True, False],
            "tol": loguniform(1e-5, 1e-2),
            "warm_start": [True, False],
            "momentum": np.linspace(0.1, 0.9, 20),
            "nesterovs_momentum": [True, False],
            "early_stopping": [True, False],
            "validation_fraction": loguniform(0.1, 0.3),
            "beta_1": np.linspace(0.1, 0.9, 20),
            "beta_2": np.linspace(0.1, 0.9, 20),
            "epsilon": loguniform(1e-8, 1e-4),
            "n_iter_no_change": [10, 20, 30],
            "max_fun": [15000],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(mlp_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning MLPRegressor model: {e}")
        return None, None
        
@st.cache
def knn_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        knn_regressor = KNeighborsRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30, 40, 50],
            "p": [1, 2],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(knn_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning KNeighborsRegressor model: {e}")
        return None, None
    
@st.cache
def extra_trees_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        extra_trees_regressor = ExtraTreesRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "criterion": ["mse", "mae", "poisson"],
            "max_depth": [10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "min_weight_fraction_leaf": uniform(0.0, 0.5),
            "max_features": ["auto", "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 20, 30],
            "min_impurity_decrease": uniform(0.0, 0.5),
            "bootstrap": [True, False],
            "oob_score": [True, False],
            "random_state": [None, 42],
            "warm_start": [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(extra_trees_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning ExtraTreesRegressor model: {e}")
        return None, None
        
@st.cache
def kernel_ridge_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        kernel_ridge = KernelRidge()

        # Define hyperparameters to tune
        param_dist = {
            "alpha": uniform(1e-6, 1.0),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": randint(1, 10),
            "gamma": ["scale", "auto", uniform(0.001, 1)],
            "coef0": uniform(-1, 1),
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(kernel_ridge, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning KernelRidge model: {e}")
        return None, None
    
@st.cache
def ada_boost_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        ada_boost_regressor = AdaBoostRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "base_estimator": [None, LinearRegression(), Ridge(), BayesianRidge()],
            "n_estimators": [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "learning_rate": uniform(0.1, 1.0),
            "loss": ["linear", "square", "exponential"],
            "random_state": [42],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(ada_boost_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning AdaBoostRegressor model: {e}")
        return None, None
        
@st.cache
def passive_aggressive_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        passive_aggressive_regressor = PassiveAggressiveRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "C": uniform(0.1, 10),
            "fit_intercept": [True, False],
            "max_iter": [1000, 2000, 3000],
            "tol": uniform(1e-5, 1e-2),
            "early_stopping": [True, False],
            "validation_fraction": uniform(0.1, 0.3),
            "n_iter_no_change": [5, 10, 15],
            "shuffle": [True, False],
            "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
            "random_state": [42],
            "warm_start": [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(passive_aggressive_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning PassiveAggressiveRegressor model: {e}")
        return None, None
        
@st.cache
def gradient_boosting_regressor_hyperparam_search(X_train, y_train):
    try:
        # Define the model
        gradient_boosting_regressor = GradientBoostingRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "loss": ["ls", "lad", "huber", "quantile"],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "subsample": uniform(0.1, 1.0),
            "criterion": ["friedman_mse", "mse", "mae"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "min_weight_fraction_leaf": uniform(0.0, 0.5),
            "max_depth": [3, 4, 5, 6, 7],
            "min_impurity_decrease": uniform(0.0, 0.5),
            "max_features": ["auto", "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 20, 30],
            "warm_start": [True, False],
            "validation_fraction": uniform(0.1, 0.3),
            "n_iter_no_change": [5, 10, 15],
            "tol": uniform(1e-5, 1e-2),
            "random_state": [42],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(gradient_boosting_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning GradientBoostingRegressor model: {e}")
        return None, None
        
@st.cache
def tune_sgd_regressor(X_train, y_train):
    try:
        # Define the model
        sgd_regressor = SGDRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            "penalty": ["l2", "l1", "elasticnet"],
            "alpha": uniform(1e-6, 1.0),
            "l1_ratio": uniform(0.01, 0.99),
            "fit_intercept": [True, False],
            "max_iter": [1000, 2000, 3000],
            "tol": uniform(1e-5, 1e-2),
            "shuffle": [True, False],
            "epsilon": uniform(0.1, 1.0),
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "eta0": uniform(0.01, 1.0),
            "power_t": uniform(0.1, 1.0),
            "early_stopping": [True, False],
            "validation_fraction": uniform(0.1, 0.3),
            "n_iter_no_change": [5, 10, 15],
            "warm_start": [True, False],
            "average": [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(sgd_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning SGDRegressor model: {e}")
        return None, None
    
@st.cache
def tune_rf_regressor(X_train, y_train):
    try:
        # Define the model
        rf_regressor = RandomForestRegressor()

        # Define hyperparameters to tune
        param_dist = {
            "n_estimators": [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            "criterion": ["mse", "mae"],
            "max_depth": [10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "min_weight_fraction_leaf": uniform(0.0, 0.5),
            "max_features": ["auto", "sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 20, 30],
            "min_impurity_decrease": uniform(0.0, 0.5),
            "bootstrap": [True, False],
            "oob_score": [True, False],
            "warm_start": [True],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(rf_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning RandomForestRegressor model: {e}")
        return None, None
        
@st.cache
def tune_hist_gradient_boosting_regressor(X_train, y_train):
    try:
        # Define the model
        hist_gbr = HistGradientBoostingRegressor()

        # Define hyperparameters to tune
        param_dist = {
            'loss': ['squared_error', 'absolute_error', 'gamma', 'poisson'],
            'learning_rate': uniform(0.01, 0.5),
            'max_iter': randint(50, 500),
            'max_leaf_nodes': randint(10, 100),
            'max_depth': [None] + list(randint(2, 20)),
            'min_samples_leaf': randint(5, 50),
            'l2_regularization': uniform(0, 1),
            'max_features': uniform(0.1, 1.0),
            'max_bins': randint(32, 256),
            'categorical_features': ['auto', 'from_dtype', None],
            'early_stopping': ['auto', True, False],
            'n_iter_no_change': randint(5, 20),
            'validation_fraction': uniform(0.1, 0.3),
            'tol': [1e-8, 1e-7, 1e-6, 1e-5],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(hist_gbr, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_

    except Exception as e:
        st.error(f"An error occurred while tuning HistGradientBoostingRegressor model: {e}")
        return None, None
        
@st.cache
def tune_bagging_regressor(X_train, y_train):
    try:
        # Define the model
        bagging_regressor = BaggingRegressor()

        # Define hyperparameters to tune
        param_dist = {
            'n_estimators': randint(10, 200),
            'max_samples': uniform(0.1, 1.0),
            'max_features': uniform(0.1, 1.0),
            'bootstrap': [True, False],
            'bootstrap_features': [True, False],
            'oob_score': [True, False],
            'warm_start': [True, False],
        }

        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(bagging_regressor, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
        random_search.fit(X_train, y_train)

        # Best parameters and best score
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)

        return random_search.best_estimator_, random_search.best_params_
    
    except Exception as e:
        st.error(f"An error occurred while tuning BaggingRegressor model: {e}")
        return None, None
        
@st.cache
def tune_lgbm_regressor(X_train, y_train):
    try:
        
        # Define the model
        lgbm_regressor = LGBMRegressor()
        
        # Define hyperparameters to tune
        param_dist = {
            'boosting_type': ['gbdt', 'dart', 'rf'],
            'num_leaves': randint(10, 200),
            'max_depth': randint(-1, 20),
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'n_estimators': randint(100, 1000),
            'subsample_for_bin': randint(20000, 300000),
            'min_split_gain': uniform(0, 1),
            'min_child_weight': uniform(1e-5, 1e-1),
            'min_child_samples': randint(5, 100),
            'subsample': uniform(0.5, 1),
            'subsample_freq': randint(0, 10),
            'colsample_bytree': uniform(0.5, 1),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
            'random_state': randint(1, 1000)
        }
        
        # Perform RandomizedSearchCV
        random_search = RandomizedSearchCV(lgbm_regressor, param_distributions=param_dist,
                                        n_iter=100, cv=5, random_state=42, n_jobs=-1)
        random_search.fit(X_train, y_train)
        
        # Best parameters and best score
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        return best_model, best_params
    
    except Exception as e:
        st.error(f"An error occurred while tuning LGBMRegressor model: {e}")
        return None, None
        
@st.cache
def tune_xgb_regressor(X_train, y_train):
    try:

        # Define the parameter grid for the search
        param_grid = {
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [i for i in range(100, 1400, 150)],
            'subsample': [i/10 for i in range(1, 10)],
            'colsample_bytree': [i/10 for i in range(1, 10)],
            'reg_alpha': [i/10 for i in range(1, 10)],
            'reg_lambda': [i/10 for i in range(1, 10)],
            'min_child_weight': [i for i in range(2, 8)]
        }
        
        # Initialize the XGBoost model
        xgb_model = xgb.XGBRegressor()
        
        # Perform the randomized search with cross-validation
        search = RandomizedSearchCV(xgb_model, param_grid, cv=3, n_iter=10, random_state=42)
        search.fit(X_train, y_train)
        
        # Get the best model, parameters, and score
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        return best_model, best_params
    
    except Exception as e:
        st.error(f"An error occurred while tuning XGBRegressor model: {e}")
        return None, None

@st.cache
def evaluate_model(best_model, X_test, y_test):
    """
    Evaluate the best model on the test set using various metrics.
    Args:
        best_model: The best model to be evaluated.
        X_test (pd.DataFrame): The test set features.
        y_test (pd.Series): The test set target variable.
    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for the model.
        dict: A dictionary containing the model evaluation results.
    """
    try:
        # Create a DataFrame to store the evaluation results
        results = pd.DataFrame(columns=["Model", "MAE", "MSE", "RMSE", "R2", "RPD"])

        # Predict the target variable for the test set
        y_test_pred = best_model.predict(X_test)

        # Calculate MAE, MSE, RMSE, R2, etc.
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_test_pred)
        rpd = y_test.std() / rmse

        # Append the results to the dataframe
        results = results.append({"Model": "Best Model",
                                  "MAE": mae,
                                  "MSE": mse,
                                  "RMSE": rmse,
                                  "R2": r2,
                                  "RPD": rpd}, ignore_index=True)

        # Store the model evaluation results in the dictionary
        model_evaluation = {
            "y_test": y_test,
            "y_test_pred": y_test_pred
        }

        return results, model_evaluation

    except Exception as e:
        st.error(f"An error occurred while evaluating the model: {e}")
        return pd.DataFrame(), {}

@st.cache
def plot_scatter_subplot(model_evaluation):
    try:
        fig = go.Figure()

        # Scatter plot
        scatter_trace = go.Scatter(x=model_evaluation["y_test_pred"],
                                   y=model_evaluation["y_test"],
                                   mode='markers',
                                   marker=dict(color='blue', line=dict(color='black', width=1)),
                                   name="Predictions vs True Values")

        # Reference line
        reference_line = go.Scatter(x=[min(model_evaluation["y_test"]), max(model_evaluation["y_test"])],
                                    y=[min(model_evaluation["y_test"]), max(model_evaluation["y_test"])],
                                    mode='lines',
                                    line=dict(color='black', dash='dash'),
                                    showlegend=False)

        fig.add_trace(scatter_trace)
        fig.add_trace(reference_line)

        fig.update_xaxes(title_text="Predictions (test set)")
        fig.update_yaxes(title_text="True Values (test set)")

        fig.update_layout(title_text="Scatter Subplot")

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting scatter subplot: {e}")

@st.cache
def plot_feature_importance(best_model, X_train, y_train):
    try:
        if hasattr(best_model, 'feature_importances_'):  # For models with feature_importances_
            importances = best_model.feature_importances_
        else:  # For models without feature_importances_
            result = permutation_importance(best_model, X_train, y_train, n_repeats=10, random_state=42)
            importances = result.importances_mean

        indices = np.argsort(importances)[::-1]
        names = [X_train.columns[i] for i in indices]
        importance_values = [importances[i] for i in indices]

        fig = go.Figure(go.Pie(labels=names, values=importance_values,
                                textinfo='label+percent', hole=0.3,
                                marker=dict(colors=plt.cm.tab20c.colors, line=dict(color='white', width=2))))

        fig.update_layout(title_text="Feature Importance", margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting feature importance: {e}")
@st.cache
def plot_pdp(best_model, X_train, features, target_column):
    try:
        colors = ['#2a9d8f', '#e76f51', '#f4a261', '#738bd7', '#d35400', '#a6c7d8']
        num_features = len(features)
        max_plots_per_row = 5
        num_rows = (num_features + max_plots_per_row - 1) // max_plots_per_row
        num_cols = min(num_features, max_plots_per_row)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), constrained_layout=True)

        for j, selected_feature in enumerate(features):
            row_idx = j // max_plots_per_row
            col_idx = j % max_plots_per_row

            features_info = {
                "features": [selected_feature],
                "kind": "average",
            }

            if num_rows == 1:  # If only one row, axs is 1D
                display = PartialDependenceDisplay.from_estimator(
                    best_model,
                    X_train,
                    **features_info,
                    ax=axs[col_idx],
                )
            else:
                display = PartialDependenceDisplay.from_estimator(
                    best_model,
                    X_train,
                    **features_info,
                    ax=axs[row_idx, col_idx],
                )

            color_idx = j % len(colors)
            axs[row_idx, col_idx].set_facecolor(colors[color_idx])
            axs[row_idx, col_idx].set_title(f"PDP for {selected_feature}")
            axs[row_idx, col_idx].set_xlabel(selected_feature)
            axs[row_idx, col_idx].set_ylabel(f"Partial Dependence for {target_column}")

        fig.suptitle(f"Partial Dependence of {target_column} on Selected Features", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while plotting partial dependence: {e}")
