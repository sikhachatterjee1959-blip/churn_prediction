
from pathlib  import Path 
    
# Root directory
BASE_DIR :Path = Path(__file__).resolve().parents[1]
    
#Data  and model Path
DATA_DIR: Path = Path(BASE_DIR)/"data"
DATA_PATH:Path = Path(DATA_DIR)/"Telco_Customer_Churn.csv"
CLEANED_DATA_PATH:Path = Path(DATA_DIR)/"cleaned_data.csv"
MODEL_DIR:Path = Path(BASE_DIR)/"models"
BEST_MODEL_PATH:Path = Path(MODEL_DIR)/"best_model_joblib"
FEATURES_PATH:Path = Path(MODEL_DIR)/"feature_columns.jason"


TARGET_COL:str = "Churn"

#Train , Test and cv
TEST_SIZE:float = 0.2
RANDOM_STATE:int = 42
CV_FOLD:int= 5 
N_JOBS:int = -1
SCORING:str = "roc_auc"
print("Configurationare successfully applied")                             