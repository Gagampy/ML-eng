from pathlib import Path

DATAFOLDER_LOAD_PATH: Path = Path('/home/gagampy/EPAM/ML-eng-HW/lasso/ML-eng/2-RTN/data/split')
DATAFOLDER_SAVE_PATH: Path = Path('/home/gagampy/EPAM/ML-eng-HW/lasso/ML-eng/2-RTN/data/filtered')

RANDOM_SEED: int = 42

TRAIN_RATIO: float = 0.7
VALID_RATIO: float = 0.15

UPPER_QUANTILE: float = 0.99
LOWER_QUANTILE: float = 0.01