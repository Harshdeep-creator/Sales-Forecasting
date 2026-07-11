import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

# Define forecast_horizon used consistently across models
forecast_horizon = 3

def create_xgb_features(data_series, forecast_horizon_val):
    xgb_df_local = data_series.reset_index()
    xgb_df_local.columns = ['Date', 'Sales']
    xgb_df_local['Date'] = pd.to_datetime(xgb_df_local['Date'])
    xgb_df_local = xgb_df_local.set_index('Date')

    xgb_df_local['Month'] = xgb_df_local.index.month
    xgb_df_local['Quarter'] = xgb_df_local.index.quarter
    xgb_df_local['Season_Spring'] = (xgb_df_local.index.month.isin([3, 4, 5])).astype(int)
    xgb_df_local['Season_Summer'] = (xgb_df_local.index.month.isin([6, 7, 8])).astype(int)
    xgb_df_local['Season_Autumn'] = (xgb_df_local.index.month.isin([9, 10, 11])).astype(int)
    xgb_df_local['Season_Winter'] = (xgb_df_local.index.month.isin([12, 1, 2])).astype(int)

    for i in range(1, forecast_horizon_val + 3):
        xgb_df_local[f'Lag_{i}'] = xgb_df_local['Sales'].shift(i)

    xgb_df_local['Rolling_Mean_3'] = xgb_df_local['Sales'].rolling(window=3).mean().shift(1)
    xgb_df_local['Rolling_Std_3'] = xgb_df_local['Sales'].rolling(window=3).std().shift(1)
    return xgb_df_local.dropna()

def get_xgb_future_forecast(xgb_model, xgb_df_with_features, forecast_horizon_val):
    current_sales_series = xgb_df_with_features['Sales'].copy()
    future_dates = pd.date_range(start=xgb_df_with_features.index.max() + pd.DateOffset(months=1), periods=forecast_horizon_val, freq='ME')
    future_features = pd.DataFrame(index=future_dates)
    future_features['Month'] = future_features.index.month
    future_features['Quarter'] = future_features.index.quarter
    future_features['Season_Spring'] = (future_features.index.month.isin([3, 4, 5])).astype(int)
    future_features['Season_Summer'] = (future_features.index.month.isin([6, 7, 8])).astype(int)
    future_features['Season_Autumn'] = (future_features.index.month.isin([9, 10, 11])).astype(int)
    future_features['Season_Winter'] = (future_features.index.month.isin([12, 1, 2])).astype(int)

    xgb_forecast_values = []
    for i in range(forecast_horizon_val):
        current_date = future_dates[i]
        current_future_row = future_features.iloc[i:i+1].copy()

        for lag_num in range(1, forecast_horizon_val + 3):
            if lag_num <= len(current_sales_series):
                current_future_row[f'Lag_{lag_num}'] = current_sales_series.iloc[-lag_num]
            else:
                current_future_row[f'Lag_{lag_num}'] = current_sales_series.iloc[0] # Fallback

        if len(current_sales_series) >= 3:
            current_future_row['Rolling_Mean_3'] = current_sales_series.iloc[-3:].mean()
        else:
            current_future_row['Rolling_Mean_3'] = current_sales_series.mean() # Fallback

        current_future_row['Rolling_Std_3'] = xgb_df_with_features['Rolling_Std_3'].iloc[-1] # Simplistic, take last known std


        expected_features = xgb_model.feature_names_in_ if hasattr(xgb_model, 'feature_names_in_') else current_future_row.columns
        for col in expected_features:
            if col not in current_future_row.columns:
                current_future_row[col] = 0 # Or appropriate default
        current_future_row = current_future_row[expected_features]

        next_forecast = xgb_model.predict(current_future_row)[0]
        xgb_forecast_values.append(next_forecast)

        current_sales_series = pd.concat([current_sales_series, pd.Series([next_forecast], index=[current_date])])

    return pd.Series(xgb_forecast_values, index=future_dates, name='XGBoost Future Forecast')

@st.cache_data # Cache data loading for performance
def load_data():
    df = pd.read_csv('train.csv')

    # Parse 'Order Date' and 'Ship Date' columns to datetime objects
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')

    # Extract time features from 'Order Date'
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Week Number'] = df['Order Date'].dt.isocalendar().week.astype(int)
    df['Order Day of Week'] = df['Order Date'].dt.dayofweek # Monday=0, Sunday=6
    df['Order Quarter'] = df['Order Date'].dt.quarter

    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    df['Order Season'] = df['Order Date'].apply(get_season)
    return df

@st.cache_data # Cache expensive computations
def preprocess_data(df_raw):
    df = df_raw.copy()
    # Aggregate monthly sales (for time series and general overview)
    monthly_sales = df.set_index('Order Date')['Sales'].resample('ME').sum().fillna(0)

    # Aggregate weekly sales (for anomaly detection)
    weekly_sales = df.set_index('Order Date')['Sales'].resample('W').sum().fillna(0)
    weekly_sales_df = weekly_sales.to_frame(name='Sales')
    return monthly_sales, weekly_sales_df

@st.cache_resource # Cache model training
def train_xgboost_overall(monthly_sales_data, forecast_horizon_val):
    full_xgb_df = create_xgb_features(monthly_sales_data, forecast_horizon_val)
    if full_xgb_df.empty:
        return None, 0, 0, None

    X_full_xgb_overall = full_xgb_df.drop('Sales', axis=1)
    y_full_xgb_overall = full_xgb_df['Sales']

    # Split into train/test for metric calculation
    X_train_xgb = X_full_xgb_overall.iloc[:-forecast_horizon_val]
    y_train_xgb = y_full_xgb_overall.iloc[:-forecast_horizon_val]
    X_test_xgb = X_full_xgb_overall.iloc[-forecast_horizon_val:]
    y_test_xgb = y_full_xgb_overall.iloc[-forecast_horizon_val:]

    xgb_model_full = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model_full.fit(X_full_xgb_overall, y_full_xgb_overall, verbose=False)

    # Make predictions on the test set for evaluation
    xgb_pred_on_test_data = pd.Series(xgb_model_full.predict(X_test_xgb), index=y_test_xgb.index)

    mae_xgb = mean_absolute_error(y_test_xgb, xgb_pred_on_test_data)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_xgb, xgb_pred_on_test_data))

    return xgb_model_full, mae_xgb, rmse_xgb, full_xgb_df # Return full_xgb_df for future feature creation

@st.cache_resource
def perform_anomaly_detection(weekly_sales_df_input):
    weekly_sales_df = weekly_sales_df_input.copy()
    iso_forest = IsolationForest(random_state=42, contamination=0.05)
    iso_forest.fit(weekly_sales_df[['Sales']])
    weekly_sales_df['Anomaly_IF'] = iso_forest.predict(weekly_sales_df[['Sales']])

    window_size = 4
    weekly_sales_df['Rolling_Mean'] = weekly_sales_df['Sales'].rolling(window=window_size).mean()
    weekly_sales_df['Rolling_Std'] = weekly_sales_df['Sales'].rolling(window=window_size).std()
    z_score_threshold = 1.0
    weekly_sales_df['Z_Score'] = (weekly_sales_df['Sales'] - weekly_sales_df['Rolling_Mean']) / weekly_sales_df['Rolling_Std']
    weekly_sales_df['Anomaly_ZScore'] = ((weekly_sales_df['Z_Score'] > z_score_threshold) |
                                   (weekly_sales_df['Z_Score'] < -z_score_threshold)).astype(int)

    anomalies_if = weekly_sales_df[weekly_sales_df['Anomaly_IF'] == -1]
    anomalies_zscore = weekly_sales_df[weekly_sales_df['Anomaly_ZScore'] == 1]

    return weekly_sales_df, anomalies_if, anomalies_zscore

@st.cache_resource
def perform_clustering(df_raw):
    # Re-calculate features for product segmentation
    sub_category_sales_volume = df_raw.groupby('Sub-Category')['Sales'].sum().rename('Total_Sales_Volume')
    sub_category_avg_order_value = df_raw.groupby(['Sub-Category', 'Order ID'])['Sales'].sum().groupby('Sub-Category').mean().rename('Average_Order_Value')
    monthly_sub_category_sales = df_raw.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='ME')])['Sales'].sum().unstack(fill_value=0)
    sub_category_sales_volatility = monthly_sub_category_sales.std(axis=1).rename('Sales_Volatility').fillna(0)

    yearly_sub_category_sales = df_raw.groupby(['Sub-Category', df_raw['Order Date'].dt.year])['Sales'].sum().unstack(fill_value=0)
    yoy_growth = yearly_sub_category_sales.pct_change(axis=1)
    yoy_growth = yoy_growth.replace([float('inf'), -float('inf')], pd.NA)
    sub_category_avg_growth_rate = yoy_growth.mean(axis=1).rename('Sales_Growth_Rate').fillna(0)

    product_features = pd.concat([
        sub_category_sales_volume,
        sub_category_avg_order_value,
        sub_category_sales_volatility,
        sub_category_avg_growth_rate
    ], axis=1).fillna(0) # Final fillna for any edge cases

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(product_features)
    scaled_product_features = pd.DataFrame(scaled_features, columns=product_features.columns, index=product_features.index)

    optimal_k = 4 # From previous analysis in Task 6
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    product_features['Cluster'] = kmeans.fit_predict(scaled_product_features)

    cluster_labels = {
        0: 'Average Demand, Moderate Growth',
        1: 'High Volatility, Low Sales (Niche/Risky)',
        2: 'High Volume, Stable Growth (Core Products)',
        3: 'Low Volume, Stable, Moderate Growth (Emerging/Steady)'
    }
    product_features['Cluster_Label'] = product_features['Cluster'].map(cluster_labels)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_product_features)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=scaled_product_features.index)
    pca_df['Cluster_Label'] = product_features['Cluster_Label']

    return product_features, pca_df


# --- Load and Preprocess Data --- #
df_raw = load_data()
monthly_sales, weekly_sales_df = preprocess_data(df_raw)

# --- Train Models and Perform Analysis ---
xgb_model_overall, xgb_mae, xgb_rmse, xgb_full_df_features = train_xgboost_overall(monthly_sales, forecast_horizon)
weekly_sales_df_anom, anomalies_if, anomalies_zscore = perform_anomaly_detection(weekly_sales_df)
product_features, pca_df = perform_clustering(df_raw)

# --- Streamlit Application --- #
st.sidebar.title('Sales Data Analysis Dashboard')
selection = st.sidebar.radio(
    "Go to",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"]
)

if selection == "Sales Overview":
    st.title('Sales Overview Dashboard')

    # Total sales by year (bar chart)
    yearly_sales = df_raw.groupby('Order Year')['Sales'].sum().reset_index()
    st.subheader('Total Sales by Year')
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Order Year', y='Sales', data=yearly_sales, ax=ax1, palette='viridis')
    ax1.set_title('Total Sales by Year')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Sales')
    st.pyplot(fig1)

    # Monthly sales trend line chart
    st.subheader('Monthly Sales Trend (2015-2018)')
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, ax=ax2)
    ax2.set_title('Overall Monthly Sales Trend')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Total Monthly Sales')
    ax2.grid(True)
    st.pyplot(fig2)

    # Sales by region and category (interactive filters)
    st.subheader('Sales by Region and Category')
    selected_region = st.selectbox('Select Region', ['All'] + list(df_raw['Region'].unique()))
    selected_category = st.selectbox('Select Category', ['All'] + list(df_raw['Category'].unique()))

    filtered_df = df_raw.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]

    # Group by month for line plot
    filtered_monthly_sales = filtered_df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().fillna(0)

    if not filtered_monthly_sales.empty:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.lineplot(x=filtered_monthly_sales.index, y=filtered_monthly_sales.values, ax=ax3)
        ax3.set_title(f'Monthly Sales for {selected_category} in {selected_region}')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Total Monthly Sales')
        ax3.grid(True)
        st.pyplot(fig3)
    else:
        st.write("No data available for the selected filters.")

elif selection == "Forecast Explorer":
    st.title('Sales Forecast Explorer (XGBoost)')

    if xgb_model_overall is None:
        st.error("XGBoost model could not be trained due to insufficient data. Please ensure 'train.csv' has enough data points.")
    else:
        forecast_type = st.radio("Select Forecast Type", ['Overall', 'Category', 'Region'])

        segment_data = monthly_sales # Default to overall
        segment_name = "Overall Sales"

        if forecast_type == 'Category':
            selected_segment_val = st.selectbox('Select Category for Forecast', df_raw['Category'].unique())
            segment_df_filtered = df_raw[df_raw['Category'] == selected_segment_val].copy()
            segment_data = segment_df_filtered.set_index('Order Date')['Sales'].resample('ME').sum().fillna(0)
            segment_name = f"Category: {selected_segment_val}"
        elif forecast_type == 'Region':
            selected_segment_val = st.selectbox('Select Region for Forecast', df_raw['Region'].unique())
            segment_df_filtered = df_raw[df_raw['Region'] == selected_segment_val].copy()
            segment_data = segment_df_filtered.set_index('Order Date')['Sales'].resample('ME').sum().fillna(0)
            segment_name = f"Region: {selected_segment_val}"

        # Date range slider to select forecast horizon (1, 2, or 3 months ahead)
        selected_forecast_horizon = st.slider('Select Forecast Horizon (Months)', 1, 3, 3)

        if not segment_data.empty and len(segment_data) > (selected_forecast_horizon + 5): # Ensure enough data for feature engineering
            xgb_data_with_features = create_xgb_features(segment_data, selected_forecast_horizon)
            if not xgb_data_with_features.empty:
                X_full = xgb_data_with_features.drop('Sales', axis=1)
                y_full = xgb_data_with_features['Sales']

                temp_xgb_model = xgb.XGBRegressor(
                    objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
                    max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42
                )
                temp_xgb_model.fit(X_full, y_full, verbose=False)

                forecast = get_xgb_future_forecast(temp_xgb_model, xgb_data_with_features, selected_forecast_horizon)

                st.subheader(f'Forecasted Sales for {segment_name}')
                st.write(forecast)

                st.write("**Note:** MAE and RMSE are for the overall XGBoost model trained on all historical sales.")
                st.write(f"Overall XGBoost MAE: {xgb_mae:.2f}")
                st.write(f"Overall XGBoost RMSE: {xgb_rmse:.2f}")

                fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
                ax_forecast.plot(segment_data.index, segment_data.values, label='Historical Sales')
                ax_forecast.plot(forecast.index, forecast.values, label='Forecasted Sales', linestyle='--', color='green')
                ax_forecast.set_title(f'Sales Forecast for {segment_name}')
                ax_forecast.set_xlabel('Date')
                ax_forecast.set_ylabel('Sales')
                ax_forecast.legend()
                ax_forecast.grid(True)
                st.pyplot(fig_forecast)

            else:
                st.write("Not enough data after feature engineering for selected forecast.")
        else:
            st.write("Not enough historical data to generate a forecast for the selected horizon. Please ensure the segment has sufficient data points.")

elif selection == "Anomaly Report":
    st.title('Anomaly Report')

    st.subheader('Weekly Sales with Anomalies Detected by Isolation Forest and Z-Score')
    fig_anomaly, ax_anomaly = plt.subplots(figsize=(18, 7))
    sns.lineplot(x=weekly_sales_df_anom.index, y=weekly_sales_df_anom['Sales'], label='Weekly Sales', color='blue', ax=ax_anomaly)
    sns.scatterplot(x=anomalies_if.index,
                    y=anomalies_if['Sales'],
                    color='red', s=100, label='Anomaly (Isolation Forest)', marker='o', alpha=0.7, ax=ax_anomaly)
    sns.scatterplot(x=anomalies_zscore.index,
                    y=anomalies_zscore['Sales'],
                    color='purple', s=100, label='Anomaly (Z-Score)', marker='X', alpha=0.7, ax=ax_anomaly)
    ax_anomaly.set_title('Comparison of Anomalies Detected by Isolation Forest and Z-Score')
    ax_anomaly.set_xlabel('Date')
    ax_anomaly.set_ylabel('Total Weekly Sales')
    ax_anomaly.grid(True)
    ax_anomaly.legend()
    st.pyplot(fig_anomaly)

    st.subheader('Detected Anomaly Dates (Isolation Forest)')
    st.dataframe(anomalies_if[['Sales']])

    st.subheader('Detected Anomaly Dates (Z-Score)')
    st.dataframe(anomalies_zscore[['Sales']])

elif selection == "Product Demand Segments":
    st.title('Product Demand Segments')

    st.subheader('Product Sub-Category Clusters (PCA-reduced)')
    fig_cluster, ax_cluster = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Label', data=pca_df, palette='viridis', s=100, alpha=0.8, ax=ax_cluster)
    ax_cluster.set_title('Product Sub-Category Clusters (PCA-reduced)')
    ax_cluster.set_xlabel('Principal Component 1')
    ax_cluster.set_ylabel('Principal Component 2')
    ax_cluster.grid(True)
    ax_cluster.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_cluster)

    st.subheader('Sub-Categories by Demand Cluster')
    cluster_df_display = product_features[['Cluster_Label']].reset_index().rename(columns={'index': 'Sub-Category'})
    st.dataframe(cluster_df_display.sort_values('Cluster_Label'))
