# Categorizing-Fault-Patterns-in-NEVs
Fault Detection in New Energy Vehicles using K-Means clustering methodology.

## Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contact](#contact)

## About the Project

- **Description**: This project focuses on identifying and categorizing fault patterns in New Energy Vehicles (NEVs) using unsupervised clustering algorithms. Early detection of faults is crucial for preventing breakdowns, reducing maintenance costs, and enhancing vehicle reliability and safety.
- **Features**:
    - Sensor Measurements: battery voltage (V), battery current (A), engine temperature (°C), motor efficiency (%), tire pressure (PSI), fuel efficiency (MPG), speed (km/h), acceleration (m/s²), driving distance (km), ambient temperature (°C), and humidity (%).
    - Operational Metrics: last service distance (km), service frequency (count), repair cost (USD), and downtime (hours).
    - Condition Indicators: road conditions (categorical: Hilly, Smooth, Bumpy) and time since last fault (days).
    - Target Variable: fault type with 4 categories: sensor malfunction, engine overheating, battery issue, and no fault
- **Technologies Used**:
    - Python 3.11
    - Jupyter Notebook
    - pandas - Data loading and manipulation
    - numpy - Numerical computations
    - scikit-learn - K-means clustering, LabelEncoder, StandardScaler
    - matplotlib - Data visualization

## Getting Started

### Prerequisites
Before running the project, have the following installed:
- Python 3.x (tested using 3.11)
- Jupyter Notebook
- pip (Python package manager)
  
Required Python Libraries:
- pandas
- numpy
- scikit-learn
- matplotlib

### Installation
1. Clone the repository
```bash 
git clone https://github.com/imagn002-glitch/Categorizing-Fault-Patterns-in-NEVs.git
cd Categorizing-Fault-Patterns-in-NEVs
```

2. Install required packages
```bash
pip install pandas numpy scikit-learn matplotlib
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ziya07/fault-prediction-and-optimization-in-nevs/data) and place `Fault_nev_dataset.csv` in the project directory.

## Usage
1. Open Jupyter Notebook
```bash
jupyter notebook
```

2. Open the project notebook and run cells sequentially:
  - Load the dataset
  - Encode the categorical variables using LabelEncoder
  - Apply StandardScaler normalization to all numerical features
  - Execute K-means clustering algorithm
  - Visualize and analyze fault pattern clusters

Example:
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv(’Fault_nev_dataset.csv’)

# Encode categorical columns
for col in df.select_dtypes(include=’object’):
    df[col] = LabelEncoder().fit_transform(df[col])

# Feature scaling
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

## Contact
- Contributor(s): Isabel Magnuson
- Email: imagn002@ucr.edu

### Getting Help
- Email the maintainer: imagn002@ucr.edu
- Visit the [Kaggle] dataset page (https://www.kaggle.com/datasets/ziya07/fault-prediction-and-optimization-in-nevs/data) for dataset questions
