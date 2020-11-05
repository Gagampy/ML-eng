from pathlib import Path

DATAFOLDER_LOAD_PATH: Path = Path('/usr/local/rtn-project/data/split')
DATAFOLDER_SAVE_PATH: Path = Path('/usr/local/rtn-project/data/filtered')

RANDOM_SEED: int = 42

TRAIN_RATIO: float = 0.7
VALID_RATIO: float = 0.15

UPPER_QUANTILE: float = 0.99
LOWER_QUANTILE: float = 0.01
