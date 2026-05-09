import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load credit card fraud dataset from CSV."""
    path = Path(filepath)

    if not path.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic validation checks on the dataset."""
    logger.info("Running data validation...")

    # Check required columns
    required_cols = ["Time", "Amount", "Class"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("No null values found ✅")

    # Check class distribution
    fraud_count = df["Class"].sum()
    total = len(df)
    fraud_pct = (fraud_count / total) * 100
    logger.info(f"Fraud cases: {fraud_count:,} / {total:,} ({fraud_pct:.2f}%)")
    logger.info(f"Legit cases: {total - fraud_count:,} / {total:,} ({100 - fraud_pct:.2f}%)")

    return df


if __name__ == "__main__":
    # Quick test — update path to your downloaded CSV
    df = load_data("src/data/creditcard.csv")
    df = validate_data(df)
    print(df.head())
