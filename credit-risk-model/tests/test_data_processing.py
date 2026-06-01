import pytest
import pandas as pd
import numpy as np
from src.data_processing import DateFeatureExtractor, CustomerAggregateEngineer

def test_date_feature_extractor_columns():
    """Verifies that the DateFeatureExtractor creates expected sub-time elements."""
    extractor = DateFeatureExtractor(date_col='TransactionStartTime')
    
    # Create sample transactional entry matching raw layout
    sample_df = pd.DataFrame({
        'TransactionStartTime': ['2026-06-01 11:24:07']
    })
    
    transformed_df = extractor.transform(sample_df)
    
    # Assert newly appended date/time variables exist
    assert 'TransactionHour' in transformed_df.columns
    assert 'TransactionDay' in transformed_df.columns
    assert 'TransactionMonth' in transformed_df.columns
    assert 'TransactionYear' in transformed_df.columns
    assert transformed_df['TransactionHour'].iloc[0] == 11

def test_customer_aggregate_engineer_output():
    """Verifies that the aggregator maps values down to the unique customer level."""
    aggregator = CustomerAggregateEngineer(customer_id_col='CustomerId', amount_col='Amount')
    
    # Mock multiple transactional rows for a single user tracking spend
    sample_data = pd.DataFrame({
        'CustomerId': ['CustomerId_1', 'CustomerId_1'],
        'Amount': [100.0, 200.0],
        'ProductCategory': ['Airtime', 'Airtime'],
        'ChannelId': ['Web', 'Web']
    })
    
    processed_df = aggregator.transform(sample_data)
    
    # Assert data was compressed into 1 single user row entry
    assert len(processed_df) == 1
    assert 'Total_Transaction_Amount' in processed_df.columns
    assert processed_df['Total_Transaction_Amount'].iloc[0] == 300.0