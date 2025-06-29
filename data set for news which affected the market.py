from gettext import install


pip install datasets
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("sohomghosh/BASIR_Budget_Assisted_Sectoral_Impact_Ranking")
