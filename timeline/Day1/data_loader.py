#Here in this we have written a DataLoader class for practice

from pathlib import Path
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self , model_name: str , filePath: str):
        self.model_name = model_name
        self.path = filePath
        self._df = None
        self._validate_path(self.path)

    def __repr__(self):
        status = f"{self._df.shape}" if self._df is not None else "not loaded"
        return f"DataLoader(model={self.model_name}, path={self.path}, data={status})"

    def __str__(self):
        status = f"{self._df.shape[0]:,} rows x {self._df.shape[1]} cols" if self._df is not None else "not loaded"
        return f"You are working on model '{self.model_name}' — data: {status}"
    
    def _validate_path(self, path: str) -> bool:
        valid_extensions = [".csv", ".json", ".txt", ".parquet"]
    
        if not any(path.endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported format choose from {valid_extensions}")
    
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")
    
        return True

    def load(self, **kwargs) -> pd.DataFrame:
        print(f"Loading {self.path} ...")
    
        if self.path.endswith(".csv"):
            self._df = pd.read_csv(self.path, **kwargs)
        
        elif self.path.endswith(".parquet"):
            self._df = pd.read_parquet(self.path, **kwargs)
        
        elif self.path.endswith(".json"):
            self._df = pd.read_json(self.path, **kwargs)
        
        elif self.path.endswith(".txt"):
            self._df = pd.read_csv(self.path, sep="\t", **kwargs)

        else:
            raise ValueError(f"Cannot load file: {self.path}")
    
        print(f"Loaded {len(self._df):,} rows x {len(self._df.columns)} columns")
    
        return self._df
    
    def validate(self) -> dict:
    
        if self._df is None:
            raise RuntimeError("Call load() before validate()")
    
        results = {}
    
        # 1. Shape
        results["shape"] = self._df.shape
    
        # 2. Null values
        null_counts = self._df.isna().sum()
        results["null_counts"] = null_counts[null_counts > 0].to_dict()
        results["null_percent"] = (
            (null_counts / len(self._df) * 100)
            .round(2)[null_counts > 0]
            .to_dict()
        )
    
        # 3. Duplicate rows
        results["duplicate_rows"] = int(self._df.duplicated().sum())
    
        # 4. Data types
        results["dtypes"] = self._df.dtypes.astype(str).to_dict()
    
        # 5. Constant columns (zero variance — useless for ML)
        constant_cols = [
            col for col in self._df.select_dtypes(include=[np.number]).columns
            if self._df[col].nunique() <= 1
        ]
        results["constant_columns"] = constant_cols
    
        # Print findings
        print(f"Shape: {results['shape']}")
    
        if results["null_counts"]:
            print(f"Columns with nulls: {results['null_counts']}")
        else:
            print("No null values found.")
        
        if results["duplicate_rows"] > 0:
            print(f"Duplicate rows: {results['duplicate_rows']}")
        else:
            print("No duplicate rows found.")
        
        if constant_cols:
            print(f"Constant columns: {constant_cols}")
        else:
            print("No constant columns found.")
    
        return results
    
    def summary(self) -> pd.DataFrame:
    
        if self._df is None:
            raise RuntimeError("Call load() before summary()")
    
        rows = []
    
        for col in self._df.columns:
            series = self._df[col]
        
            row = {
                "column"   : col,
                "dtype"    : str(series.dtype),
                "non_null" : int(series.notna().sum()),
                "null_pct" : round(series.isna().mean() * 100, 2),
                "unique"   : series.nunique(),
            }
        
            # Numeric columns get statistical measures
            if pd.api.types.is_numeric_dtype(series):
                row["mean"] = round(series.mean(), 4)
                row["std"]  = round(series.std(), 4)
                row["min"]  = series.min()
                row["25%"]  = series.quantile(0.25)
                row["50%"]  = series.quantile(0.50)
                row["75%"]  = series.quantile(0.75)
                row["max"]  = series.max()
                row["top_value"] = None
            
            # Categorical/text columns get most frequent value instead
            else:
                row["mean"] = None
                row["std"]  = None
                row["min"]  = None
                row["25%"]  = None
                row["50%"]  = None
                row["75%"]  = None
                row["max"]  = None
                row["top_value"] = series.mode().iloc[0] if not series.empty else None
        
            rows.append(row)
    
        summary_df = pd.DataFrame(rows).set_index("column")
        summary_df = summary_df.fillna("")
        
        print(summary_df.to_string())
    
        return summary_df