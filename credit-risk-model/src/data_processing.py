import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from IPython.display import display


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts date-time attributes from transaction timestamps."""
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Ensure datetimes are parsed correctly
        datetime_series = pd.to_datetime(X_out[self.date_col])
        X_out['TransactionHour'] = datetime_series.dt.hour
        X_out['TransactionDay'] = datetime_series.dt.day
        X_out['TransactionMonth'] = datetime_series.dt.month
        X_out['TransactionYear'] = datetime_series.dt.year
        return X_out


class CustomerAggregateEngineer(BaseEstimator, TransformerMixin):
    """Transforms transaction-level rows into aggregated customer features."""
    def __init__(self, customer_id_col='CustomerId', amount_col='Amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Build aggregation dictionary
        agg_funcs = {
            self.amount_col: ['sum', 'mean', 'count', 'std']
        }
        
        # Group transaction records by unique customer identifiers
        customer_aggs = X_out.groupby(self.customer_id_col).agg(agg_funcs)
        
        # Flatten multi-level indices safely
        customer_aggs.columns = [
            'Total_Transaction_Amount',
            'Average_Transaction_Amount',
            'Transaction_Count',
            'Std_Dev_Transaction_Amount'
        ]
        
        # Resolve standard deviation for single-transaction customers (NaN -> 0.0)
        customer_aggs['Std_Dev_Transaction_Amount'] = customer_aggs['Std_Dev_Transaction_Amount'].fillna(0.0)
        
        # Bring categorical preferences into customer structure using mode
        mode_features = X_out.groupby(self.customer_id_col).agg({
            'ProductCategory': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
            'ChannelId': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        })
        
        # Combine metrics and categories
        processed_customers = customer_aggs.join(mode_features).reset_index()
        return processed_customers


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Handles categorical columns via explicit, stable multi-column label encoding."""
    def __init__(self, cols_to_encode=['ProductCategory', 'ChannelId']):
        self.cols_to_encode = cols_to_encode
        self.encoding_maps_ = {}

    def fit(self, X, y=None):
        for col in self.cols_to_encode:
            if col in X.columns:
                unique_cats = X[col].unique()
                # Create a mapping dictionary keeping 0 reserved for unknown runtime elements
                self.encoding_maps_[col] = {cat: i+1 for i, cat in enumerate(unique_cats)}
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols_to_encode:
            if col in X_out.columns:
                mapping = self.encoding_maps_[col]
                # Fallback to 0 if a category wasn't present during initial fitting
                X_out[col] = X_out[col].map(mapping).fillna(0).astype(int)
        return X_out


class StandardScaleNumerical(BaseEstimator, TransformerMixin):
    """Standardizes quantitative values to mean=0 and variance=1."""
    def __init__(self, num_cols=['Total_Transaction_Amount', 'Average_Transaction_Amount', 'Transaction_Count', 'Std_Dev_Transaction_Amount']):
        self.num_cols = num_cols
        self.means_ = {}
        self.stds_ = {}

    def fit(self, X, y=None):
        for col in self.num_cols:
            if col in X.columns:
                self.means_[col] = X[col].mean()
                self.stds_[col] = X[col].std()
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.num_cols:
            if col in X_out.columns:
                mean = self.means_[col]
                std = self.stds_[col] if self.stds_[col] != 0 else 1e-6
                X_out[col] = (X_out[col] - mean) / std
        return X_out


def build_feature_engineering_pipeline():
    """Assembles and returns the full sequential feature processing pipeline."""
    feature_pipeline = Pipeline([
        ('date_extractor', DateFeatureExtractor()),
        ('customer_aggregator', CustomerAggregateEngineer()),
        ('categorical_encoder', CategoricalEncoder()),
        ('numerical_scaler', StandardScaleNumerical())
    ])
    return feature_pipeline


if __name__ == "__main__":
    # Test script execution
    import os
    
    raw_data_path = '../data/raw/data.csv'
    if os.path.exists(raw_data_path):
        print("Loading raw transactional data...")
        df_raw = pd.read_csv(raw_data_path)
        
        print("Initializing operational feature pipeline...")
        pipeline = build_feature_engineering_pipeline()
        
        print("Fitting and executing engineering transformations...")
        df_processed = pipeline.fit_transform(df_raw)
        
        print(f"Transformation complete! Shape: {df_processed.shape}")
        print("Processed Columns:", df_processed.columns.tolist())
        display(df_processed.head(3))
    else:
        print(f"Raw data file not found at local path '{raw_data_path}'. Please check project path mapping.")