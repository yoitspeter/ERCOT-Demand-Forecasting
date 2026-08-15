import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import xgboost as xgb
from meteostat import Hourly
from datetime import datetime

def load_data(
    start=datetime(2017, 1, 1),
    end=datetime(2025, 5, 31),
    station='KAXH0',
    csv_path='master.csv'
):
    """
    Import and merge ERCOT load data with weather data.

    Args:
        start (datetime): Start date for weather data.
        end (datetime): End date for weather data.
        station (str): Meteostat station ID.
        csv_path (str): Path to load CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame with weather and ERCOT load.
    """
    load_df = pd.read_csv(csv_path, index_col="Hour Ending", parse_dates=True).asfreq("h")
    load_df = load_df[['ERCOT']].dropna()

    weather = Hourly(station, start, end).fetch()
    weather = weather[['temp', 'dwpt']].asfreq('h').dropna()
    weather = weather.shift(1)
    weather['temp'] = (weather['temp'] * 9/5) + 32
    weather['dwpt'] = (weather['dwpt'] * 9/5) + 32

    df = load_df.join(weather, how='inner')
    df.index.name = 'Date'
    df = df.dropna()

    return df
 
def split_to_train_test(
    data,
    months_to_test=12
):
    """
    Splits dataset into train and test sets based on date index.

    Args:
        data (pd.DataFrame): Main dataset with datetime index.
        months_to_test (int): Number of months to reserve for testing.

    Returns:
        tuple: (train, test) DataFrames
    """
    split_date = data.index.max() - DateOffset(months=months_to_test)
    train = data.loc[:split_date]
    test = data.loc[split_date:]

    return train, test

def feature_creation(
    data
):
    """
    Creates engineered features.

    Adds:
        - Weather-based features (HDD, CDD, etc.)
        - Lagged features (1-3h and 22-24h lags)
        - Rolling means and std devs
        - Calendar-based features (hour, day, month, school days)
        - One-hot encodings for season, hour, day of week, and month

    Args:
        data (pd.DataFrame): Input data with datetime index and columns: 'temp', 'dwpt', 'ERCOT'.

    Returns:
        pd.DataFrame: DataFrame with added features.
    """

    df = data.copy()

    ### Weather features
    base_temp = 65

    df['t_feels'] = 0.5 * (df['temp'] + df['dwpt'])
    
    df['HDD'] = np.maximum(0, base_temp - df['temp'])
    df['CDD'] = np.maximum(0, df['temp'] - base_temp)

    df['temp_change'] = df['temp'] - df['temp'].shift(1)
    df['dwpt_change'] = df['dwpt'] - df['dwpt'].shift(1)
    df['temp_squared'] = df['temp'] ** 2
    df['dwpt_squared'] = df['dwpt'] ** 2

    ### Statistical Features
    df['temp_roll_6h'] = df['temp'].rolling(window=6).mean().shift(1)
    df['temp_roll_12h'] = df['temp'].rolling(window=12).mean().shift(1)
    df['temp_roll_24h'] = df['temp'].rolling(window=24).mean().shift(1)

    df['temp_std_6h'] = df['temp'].rolling(window=6).std().shift(1)
    df['temp_std_12h'] = df['temp'].rolling(window=12).std().shift(1)
    df['temp_std_24h'] = df['temp'].rolling(window=24).std().shift(1)

    df['dwpt_roll_6h'] = df['dwpt'].rolling(window=6).mean().shift(1)
    df['dwpt_roll_12h'] = df['dwpt'].rolling(window=12).mean().shift(1)
    df['dwpt_roll_24h'] = df['dwpt'].rolling(window=24).mean().shift(1)

    df['dwpt_std_6h'] = df['dwpt'].rolling(window=6).std().shift(1)
    df['dwpt_std_12h'] = df['dwpt'].rolling(window=12).std().shift(1)
    df['dwpt_std_24h'] = df['dwpt'].rolling(window=24).std().shift(1)

    df['ercot_roll_6h'] = df['ERCOT'].rolling(window=6).mean().shift(1)
    df['ercot_roll_12h'] = df['ERCOT'].rolling(window=12).mean().shift(1)
    df['ercot_roll_24h'] = df['ERCOT'].rolling(window=24).mean().shift(1)

    df['ercot_std_6h'] = df['ERCOT'].rolling(window=6).std().shift(1)
    df['ercot_std_12h'] = df['ERCOT'].rolling(window=12).std().shift(1)
    df['ercot_std_24h'] = df['ERCOT'].rolling(window=24).std().shift(1)

    ### Calendar/time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['school_day'] = df.index.to_series().apply(
        lambda dt: int(
            dt.weekday() < 5 and
            dt.month not in [6, 7, 8]
        )
    )

    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'shoulder', 4: 'shoulder', 5: 'shoulder',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'shoulder', 10: 'shoulder', 11: 'shoulder'
    }
    df['season'] = df.index.month.map(season_map)

    month_dummies = pd.get_dummies(df['month'], prefix='month')
    hour_dummies = pd.get_dummies(df['hour'], prefix='hour')
    dow_dummies = pd.get_dummies(df['day_of_week'], prefix='dow')
    season_dummies = pd.get_dummies(df['season'], prefix='season')

    df = pd.concat([df, season_dummies, month_dummies, hour_dummies, dow_dummies], axis=1)
    df = df.drop(columns=['season', 'month', 'hour', 'day_of_week'])

    ### Lagged features
    lags = [1, 2, 3, 22, 23, 24]

    for lag in lags:
        df[f'lagged_ercot_{lag}'] = df['ERCOT'].shift(lag)
        df[f'lagged_temp_{lag}'] = df['temp'].shift(lag)
        df[f'lagged_dwpt_{lag}'] = df['dwpt'].shift(lag)
        df[f'lagged_t_feels_{lag}'] = df['t_feels'].shift(lag)
    
    return df

def prepare_train_test_sets(
    train, 
    test, 
    target
):
    """
    Splits train and test DataFrames into features and target sets.

    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Testing dataset.
        target (str): Name of the target column.

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    features = [col for col in train.columns if col != target]
    
    x_train = train[features]
    y_train = train[target]
    x_test = test[features]
    y_test = test[target]

    return x_train, y_train, x_test, y_test

def train_model(
    x_train, 
    y_train, 
    x_test, 
    y_test
):
    """
    Trains an XGBoost regressor model using the provided training and testing data.

    Args:
        x_train (pd.DataFrame): Features for training.
        y_train (pd.Series or pd.DataFrame): Target variable for training.
        x_test (pd.DataFrame): Features for validation/testing.
        y_test (pd.Series or pd.DataFrame): Target variable for validation/testing.

    Returns:
        xgb.XGBRegressor: The trained XGBoost regressor model.
    """
    reg = xgb.XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        early_stopping_rounds=5,
        random_state=42
    )

    reg.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_test, y_test)],
        verbose=50
    )

    return reg

df = load_data()
train,test = split_to_train_test(df,19)
train = feature_creation(train)
test = feature_creation(test)
train = train.dropna()
test = test.dropna()
target = 'ERCOT'
x_train, y_train, x_test, y_test = prepare_train_test_sets(train, test, target)

reg = train_model(x_train, y_train, x_test, y_test)
