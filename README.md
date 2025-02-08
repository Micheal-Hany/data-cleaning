## Advanced Data Cleaning Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![CI/CD](https://github.com/yourusername/data-cleaning/actions/workflows/main.yml/badge.svg)

Automated data preprocessing pipeline for machine learning preparation with comprehensive cleaning capabilities.

## Features 🛠️
- **Data Cleaning**: Missing value imputation (median/mode/KNN), outlier detection (Z-score/IQR), duplicate removal.
- **Feature Engineering**: Date/time feature extraction, categorical encoding (One-Hot/Label), text processing (TF-IDF vectorization), polynomial feature generation.
- **Optimization**: Skew correction (Yeo-Johnson transform), dimensionality reduction (PCA), feature scaling (StandardScaler), parallel processing support.

## Installation 📦
### Requirements
- Python 3.8+
- pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-cleaning.git
   cd data-cleaning
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start 🚀
### Basic Usage
```python
from data_cleaner import AdvancedDataCleaner

# Initialize with default configuration
cleaner = AdvancedDataCleaner()

# Process data
cleaned_data = cleaner.clean("input_data.csv")

# Save results
cleaned_data.to_csv("cleaned_output.csv", index=False)
```

## Configuration Options ⚙️
| Parameter          | Type    | Default | Description                                      |
|--------------------|---------|---------|--------------------------------------------------|
| missing_threshold   | float   | 0.7     | Drop columns with >70% missing values            |
| outlier_method      | str     | 'zscore'| Outlier detection method (zscore/iqr)            |
| max_categories       | int     | 20      | Max categories for One-Hot Encoding               |
| text_features        | bool    | False   | Enable TF-IDF text processing                     |
| dimension_reduction  | bool    | False   | Enable PCA dimensionality reduction                |
| n_jobs              | int     | -1      | CPU cores to use (-1 = all cores)                |

## Configuration

The `self.config` dictionary in the data cleaning script is a crucial component that allows users to customize the data cleaning process according to their specific needs. It is a flexible and powerful tool that enables the modification of various parameters to suit different datasets and requirements. By adjusting the values in the `self.config` dictionary, users can control how the data cleaning script handles missing values, outliers, data scaling, categorical features, text processing, dimensionality reduction, and parallel processing.

Here are the parameters included in the `self.config` dictionary:

- **missing_threshold**: Set to `0.7`, this parameter defines the threshold for missing values. If the proportion of missing values in a feature exceeds this threshold, that feature will be dropped.
- **outlier_method**: Set to `'zscore'`, this specifies the method used for detecting outliers in the dataset. The Z-score method identifies outliers based on standard deviations from the mean.
- **scale_data**: A boolean value set to `True`, indicating whether to scale the data before applying transformations. Scaling is essential for algorithms sensitive to the magnitude of features.
- **max_categories**: Set to `20`, this parameter limits the maximum number of unique categories allowed in categorical features. Features with more unique categories will be handled differently to avoid high cardinality issues.
- **text_features**: A boolean value set to `False`, indicating whether to treat features as text. If set to `True`, specific text processing methods will be applied.
- **dimension_reduction**: A boolean value set to `False`, indicating whether to apply dimensionality reduction techniques to the dataset. If set to `True`, methods like PCA may be used to reduce the number of features.
- **n_jobs**: Set to `-1`, this parameter specifies the number of jobs to run in parallel for processing. A value of `-1` means using all available processors.

### Detailed Configuration Explanation

The `self.config` dictionary is a key component of the data cleaning script, allowing users to customize the cleaning process to suit their specific needs. The dictionary contains several parameters that control various aspects of the cleaning process.

#### Missing Value Handling

The `missing_threshold` parameter determines the threshold for missing values in a feature. If the proportion of missing values in a feature exceeds this threshold, the feature will be dropped. This parameter is set to `0.7` by default, meaning that features with more than 70% missing values will be dropped.

#### Outlier Detection

The `outlier_method` parameter specifies the method used for detecting outliers in the dataset. The Z-score method is used by default, which identifies outliers based on standard deviations from the mean.

#### Data Scaling

The `scale_data` parameter is a boolean value that indicates whether to scale the data before applying transformations. Scaling is essential for algorithms sensitive to the magnitude of features. This parameter is set to `True` by default.

#### Categorical Feature Handling

The `max_categories` parameter limits the maximum number of unique categories allowed in categorical features. Features with more unique categories will be handled differently to avoid high cardinality issues. This parameter is set to `20` by default.

#### Text Feature Handling

The `text_features` parameter is a boolean value that indicates whether to treat features as text. If set to `True`, specific text processing methods will be applied. This parameter is set to `False` by default.

#### Dimensionality Reduction

The `dimension_reduction` parameter is a boolean value that indicates whether to apply dimensionality reduction techniques to the dataset. If set to `True`, methods like PCA may be used to reduce the number of features. This parameter is set to `False` by default.

#### Parallel Processing

The `n_jobs` parameter specifies the number of jobs to run in parallel for processing. A value of `-1` means using all available processors. This parameter is set to `-1` by default.

## Usage Examples 📝
### Custom Configuration
```python
config = {
    'missing_threshold': 0.5,
    'outlier_method': 'iqr',
    'text_features': True,
    'max_categories': 15
}

cleaner = AdvancedDataCleaner(config)
cleaned_data = cleaner.clean("raw_dataset.csv")
```
### Command Line Interface
```bash
python cleaning_script.py \
  --input raw_data.csv \
  --output cleaned_data.csv \
  --config config.json
```

## Output Format 📄
### Sample Output Structure
```csv
age_scaled,income_normalized,gender_encoded,pc1,pc2
-0.52,1.23,0,0.12,-0.45
0.78,-0.92,1,1.22,0.31
```
### Report Format
```
[Cleaning Report]
- Removed 2 columns (50%+ missing values)
- Encoded 3 categorical features
- Detected and removed 15 outliers
- Generated 5 interaction features
- Reduced to 2 principal components
- Final dataset: 1000 rows × 8 columns
```

## Contributing 🤝
### Workflow
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-feature`.
3. Commit changes: `git commit -m 'Add awesome feature'`.
4. Push to branch: `git push origin feature/new-feature`.
5. Open a Pull Request.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Build documentation
mkdocs build
```

## License 📜
This project is licensed under the MIT License.

## Support 🆘
### Community Support
- GitHub Discussions
- Stack Overflow

### Enterprise Support
- Email: support@yourcompany.com
- Service Level Agreement: 24-hour response time

## Changelog 📌
v1.2.0 (2023-08-15)
- Added KNN imputation.
- Improved parallel processing.
- Enhanced date parsing.

v1.1.0 (2023-08-01)
- Added TF-IDF vectorization.
- Implemented PCA support.
- Improved documentation.

v1.0.0 (2023-07-15)
- Initial release.
- Core cleaning features.
- Basic configuration system.

**To Use This File:**
1. Replace placeholders (`yourusername`, `yourcompany.com`) with your actual information.
2. Save as `README.md` in your project root.
3. Create accompanying files:
   - `LICENSE.md`
   - `requirements.txt`
   - `requirements-dev.txt`
4. Add actual implementation code in `data_cleaner.py`.

This README provides complete documentation while maintaining readability and a professional presentation.
