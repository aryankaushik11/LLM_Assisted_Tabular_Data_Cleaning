import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading of raw datasets (CSV) with basic validation.
    """
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = data_path

    def load_csv(self, filename: str, encoding: str = 'utf-8', **kwargs) -> pd.DataFrame:
        """
        Loads a CSV file into a Pandas DataFrame.
        """
        file_path = os.path.join(self.data_path, filename)
        
        if not os.path.exists(file_path):
            # Check if absolute path was provided or fallback
            if os.path.exists(filename):
                file_path = filename
            else:
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Attempt to read with provided encoding
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            logger.info(f"Successfully loaded {filename} with shape {df.shape}")
            
            # Simple validation: Check if file is empty
            if df.empty:
                logger.warning(f"Warning: {filename} is empty.")
                
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            return df
            
        except UnicodeDecodeError:
            logger.warning(f"Encoding {encoding} failed. Trying 'latin1'...")
            return self.load_csv(filename, encoding='latin1', **kwargs)
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise e

    def validate_schema(self, df: pd.DataFrame, required_columns: list = None) -> bool:
        """
        Checks if required columns exist in the DataFrame.
        """
        if required_columns is None:
            return True
            
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        return True
