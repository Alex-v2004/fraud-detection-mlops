import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the fraud dataset:
    - Scale Amount and Time
    - Split into train/test
    """
    logger.info("Starting preprocessing...")

    df = df.copy()

    # Scale Amount and Time (V1-V28 are already PCA-transformed)
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"] = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount", "Time"], inplace=True)
    logger.info("Scaled Amount and Time columns ✅")

    # Features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    logger.info(f"Train fraud rate: {y_train.mean() * 100:.2f}%")
    logger.info(f"Test fraud rate:  {y_test.mean() * 100:.2f}%")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from data_loader import load_data, validate_data
    df = load_data("src/data/creditcard.csv")
    df = validate_data(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
