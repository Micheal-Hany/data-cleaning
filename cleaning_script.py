import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler, PowerTransformer
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from scipy.stats import zscore
import warnings
from concurrent.futures import ProcessPoolExecutor
import logging
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')


class AdvancedDataCleaner:
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            'missing_threshold': 0.7,
            'outlier_method': 'zscore',
            'scale_data': True,
            'max_categories': 20,
            'text_features': False,
            'dimension_reduction': False,
            'n_jobs': -1
        }
        # Update with user-provided config
        if config:
            self.config.update(config)

        self.transformer_cache = {}
        self.log = []

    def _parallel_apply(self, func, dataframes):
        """Apply function to multiple DataFrames in parallel"""
        with ProcessPoolExecutor(max_workers=os.cpu_count() if self.config['n_jobs'] == -1 else self.config['n_jobs']) as executor:
            return list(executor.map(partial(func), dataframes))

    def load_data(self, path):
        """Load data from either a directory or single file"""
        if os.path.isdir(path):
            self.log.append(f"Loading directory: {path}")
            return self._load_directory(path)
        elif os.path.isfile(path):
            self.log.append(f"Loading single file: {path}")
            return self._load_single_file(path)
        else:
            raise ValueError(
                f"Path {path} is neither a valid directory nor file")

    def _load_directory(self, directory):
        """Load all CSV files from a directory"""
        files = [os.path.join(directory, f)
                 for f in os.listdir(directory) if f.endswith('.csv')]

        if self.config['n_jobs'] != 1:
            with ProcessPoolExecutor() as executor:
                dfs = list(executor.map(pd.read_csv, files))
        else:
            dfs = [pd.read_csv(f) for f in files]

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            self.log.append(
                f"Loaded {len(dfs)} files with total {len(combined_df)} rows")
            return combined_df
        return pd.DataFrame()

    def _load_single_file(self, file_path):
        """Load single CSV file"""
        if not file_path.endswith('.csv'):
            raise ValueError("File must be a CSV")
        return pd.read_csv(file_path)

    def _handle_dates(self, df):
        """Advanced date detection and feature engineering"""
        date_cols = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    continue

        for col in date_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df = df.drop(col, axis=1)

        return df

    def _handle_missing(self, df):
        """Advanced missing value handling with multiple strategies"""
        # Drop columns with too many missing values
        missing_perc = df.isnull().mean()
        df = df.loc[:, missing_perc < self.config['missing_threshold']]

        # Smart imputation
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                if df[col].skew() > 1:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
        return df

    def _handle_outliers(self, df):
        """Robust outlier handling with multiple methods"""
        num_cols = df.select_dtypes(include=np.number).columns

        if self.config['outlier_method'] == 'zscore':
            df = df[(np.abs(zscore(df[num_cols])) < 3).all(axis=1)]
        elif self.config['outlier_method'] == 'iqr':
            for col in num_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
        return df

    def _handle_categorical(self, df):
        """Smart categorical encoding with multiple strategies"""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            n_unique = df[col].nunique()

            if n_unique == 1:
                df.drop(col, axis=1, inplace=True)
            elif n_unique < self.config['max_categories']:
                # Updated for modern scikit-learn versions
                onehot = OneHotEncoder(
                    sparse_output=False, handle_unknown='ignore')
                encoded = onehot.fit_transform(df[[col]])
                df = pd.concat([
                    df.drop(col, axis=1),
                    pd.DataFrame(encoded, columns=[
                                 f"{col}_{cat}" for cat in onehot.categories_[0]])
                ], axis=1)
                self.transformer_cache[col] = onehot
            else:
                lbl = LabelEncoder()
                df[col] = lbl.fit_transform(df[col])
                self.transformer_cache[col] = lbl

        return df

    def _handle_text(self, df):
        """Basic text processing using TF-IDF"""
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            if df[col].apply(lambda x: len(str(x).split())).mean() > 3:
                tfidf = TfidfVectorizer(max_features=100)
                tfidf_matrix = tfidf.fit_transform(df[col])
                df = pd.concat([
                    df.drop(col, axis=1),
                    pd.DataFrame(tfidf_matrix.toarray(), columns=[
                                 f"{col}_{i}" for i in range(tfidf_matrix.shape[1])])
                ], axis=1)
        return df

    def _feature_engineering(self, df):
        """Automated feature engineering"""
        # Polynomial features
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 1:
            for i in range(len(num_cols)):
                for j in range(i+1, len(num_cols)):
                    df[f'{num_cols[i]}_x_{num_cols[j]}'] = df[num_cols[i]
                                                              ] * df[num_cols[j]]
        return df

    def _transform_skew(self, df):
        """Handle skewed data using power transforms"""
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if df[col].skew() > 1 or df[col].skew() < -1:
                pt = PowerTransformer(method='yeo-johnson')
                df[col] = pt.fit_transform(df[[col]])
                self.transformer_cache[col] = pt
        return df

    def clean(self, path):
        """Main cleaning pipeline"""
        df = self.load_data(path)
        original_shape = df.shape

        df = self._handle_dates(df)
        df = self._handle_missing(df)
        df = self._handle_outliers(df)
        df = self._handle_categorical(df)

        if self.config['text_features']:
            df = self._handle_text(df)

        df = self._feature_engineering(df)
        df = self._transform_skew(df)

        # Remove low-variance features
        selector = VarianceThreshold(threshold=0.1)
        df = pd.DataFrame(selector.fit_transform(
            df), columns=df.columns[selector.get_support()])

        # Final NaN check and handling
        df = self._final_nan_check(df)

        # Dimensionality reduction
        if self.config['dimension_reduction'] and len(df.columns) > 50:
            pca = PCA(n_components=0.95)
            df = pd.DataFrame(pca.fit_transform(df))
            self.transformer_cache['pca'] = pca

        # Scaling
        if self.config['scale_data']:
            scaler = StandardScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            self.transformer_cache['scaler'] = scaler

        final_shape = df.shape
        self.log.append(
            f"Data cleaned. Original shape: {original_shape}, Final shape: {final_shape}")
        return df

    def _final_nan_check(self, df):
        """Ensure no remaining NaN values before final transformations"""
        if df.isnull().any().any():
            self.log.append("Final NaN handling with KNN imputation")
            imputer = KNNImputer(n_neighbors=5)
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        return df

    def get_report(self):
        """Generate cleaning report"""
        return "\n".join(self.log)


if __name__ == "__main__":
    config = {
        'missing_threshold': 0.8,
        'outlier_method': 'zscore',
        'text_features': True,
        'dimension_reduction': True,
        'max_categories': 20,
        'n_jobs': -1
    }

    cleaner = AdvancedDataCleaner(config)

    # Taking the file path as input from the user
    data_path = input("Please enter the path to your CSV file: ")

    cleaned_data = cleaner.clean(data_path)

    print("Cleaning Report:")
    print(cleaner.get_report())
    print("\nFirst 5 rows of cleaned data:")
    print(cleaned_data.head())
