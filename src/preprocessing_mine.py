"""_
preprocessing utilities
-Import precleaned data
-Impute Null values
-Scale data
-Encode Categorical data
-Train test splitsummary_
    """
    
    #Imports
import numpy as np
import pandas as pd
from pathlib import Path
    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src import config import DATA_DIR, CLEANED_DATA_PATH, TARGET_COL, TEST_SIZE, RANDOM_STATE
from typing import list, Tuple

# Load pre-cleaned data
def load_data(data_path = CLEANED_DATA_PATH):
    """
    This will load pre-cleaned data.
    Remove leading or lagging white-space from column name
    Return clean data
    
    """
    df = pd.read_csv(data_path)
    
    # Normalize column columns
    df.columns = [str(c).strip() for c in df.columns.to_list()]
    return df

# Extract categorical and Numeric columns
def _extract_cat_cols_num_cols(df):
    cat_cols = df.select_dytypes(include=[""object","category","bool"]).columns.to_list()
    num_cols = df.select_dytypes(include=["int64","float64"]).columns.to_list()
    
    return cat_cols, num_cols
    
    # Build preprocessor
    def build_preprocessor(df or X: pd.DataFrame):
    
    # Create a back up
    df = df_or_X.copy()
    
    # If target column is present, drop the column
    if TARGET_COL in df.columns:
        df.drop(columns=[TARGET_COL]
    else:
        df = df
  
  # EXTRACT Numneric and Caterogical columns  
    cat_cols, num_cols = _extract_cat_cols_num_cols (df)
    
    
    # NUmeric pipeline: Impute -> Scale
    num_transformer = Pipeline(steps =[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
        
        
        # Categorical pipeline: Impute ->OneHotEncoding
        cat_transformer = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), sparse_output=False))
            
            
            
        
        ])
    
         # Combine the pipelines
         transformers  = []
         if cat_cols:
                transformers.append(("cat", cat_transformer, cat_cols))
        if num_cols:
                transformers.append(("num", num_transformer, num_cols))
        preprocessor = ColumnTransformer(transformers=transformers, remainder = "drop",verbose_freature_names_out=False)
        
        return preprocessor
        
        # Train test split
        def split_data(df:pd.DataFrame):
             df = df.copy()
         # If target column is missing
         if TARGET_COL not in df.columns:
               raise KeyError(f"Target column {TARGET_COL} is not found in DataFrame columns:{df.columns.to_list()}")
               
               X = df.drop(columns=[TARGET_COL])
               y = df[TARGET_COL]
               
               
               X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
                return X_train, X_test, y_train, y_test
                
                print("Preprocessing is executed successfully")
               
                                                                                                  
        
        
    )
    

    Args:
        data_path (Path): Path to the cleaned data CSV file.
    
     