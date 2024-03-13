import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.weight': 'normal'})

def run_eda(df: pd.DataFrame, category_values: int = 0) -> None:
    '''
    Perform Exploratory Data Analysis (EDA) on the given DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.
    - category_values (int): The threshold for unique values to consider a column as categorical.
    Columns with unique values less than or equal to this threshold will be categorized.

    Returns:
    None
    '''
    # Greeting
    print('Praise the Omnissiah! Welcome to the Sanctum of Exploratory Data Analysis.\n')
    
    # Number of observations and parameters
    num_observations = df.shape[0]
    num_parameters = df.shape[1]
    print(f'Number of Observations (Rows): {num_observations}')
    print(f'Number of Parameters (Columns): {num_parameters}')
    
    # Data types of each column
    print('\nData Types of Each Column:')
    for column_name in df.columns:
        unique_values = df[column_name].nunique()
        if unique_values <= category_values:
            df[column_name] = df[column_name].astype('category')
    column_types = df.dtypes
    max_len = max(map(len, column_types.index)) + 2
    for column_name, data_type in column_types.items():
        print(f'{column_name.ljust(max_len)} {data_type}')
    
    # Categorize features into numerical, string, and categorical
    print()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    string_features = df.select_dtypes(include=['object']).columns
    categorical_features = df.select_dtypes(include=['category']).columns
    print(f'Numerical features: {", ".join(numerical_features) if not numerical_features.empty else 0}')
    print(f'String features: {", ".join(string_features) if not string_features.empty else 0}')
    print(f'Categorical features: {", ".join(categorical_features) if not categorical_features.empty else 0}')
    
    # Counts and frequencies for categorical features
    print('\nCounts and Frequencies for Categorical Features:')
    for col in categorical_features:
        counts = df[col].value_counts()
        frequencies = counts / num_observations
        count_df = pd.DataFrame({'count': counts, 'Frequency': frequencies})
        count_df.index.name = col
        count_df = count_df.sort_index()
        print(count_df.to_string())
        
    # Descriptive statistics for numerical features except ID
    print('\nDescriptive Statistics for Numerical Features:')
    numerical_features_selected = numerical_features[~numerical_features.str.contains('Id')]
    numerical_statistics_selected = df.loc[:, numerical_features_selected].describe().round(2)
    print(numerical_statistics_selected)

    print('\nHistograms with Boxplots for Numerical Features:')
    numerical_features_selected = numerical_features[~numerical_features.str.contains('Id')]
    for col in numerical_features_selected:
        plt.figure(figsize=(6, 3))
        sns.set(style="whitegrid")
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], bins=30, kde=False, color='red')
        plt.title(f'Histogram of {col}', fontsize=10, fontweight='bold')
        plt.xlabel(col, fontsize=8, fontweight='bold')
        plt.ylabel('Count', fontsize=8, fontweight='bold')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(False)
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col], color='red')
        plt.title(f'Boxplot of {col}', fontsize=8, fontweight='bold')
        plt.xlabel(col, fontsize=6, fontweight='bold')
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.grid(False)
        plt.show()

    print('\nCorrelation Heatmap:')
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1)
    sns.heatmap(df[numerical_features_selected].corr(), annot=True, cmap='inferno', fmt=".2f")
    plt.title('Correlation Heatmap', fontsize=10, fontweight='bold')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()
    
    # Outliers for numerical features except ID
    print('\nOutliers for Numerical Features:')
    outliers = {}
    for col in numerical_features_selected:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        num_outliers = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        outliers[col] = num_outliers
    for col, num_outliers in outliers.items():
        print(f'{col}: {num_outliers}')

    # Missing values
    print('\nMissing Values:')
    total_missing = df.isnull().sum().sum()
    rows_with_missing = df[df.isnull().any(axis=1)].shape[0]
    columns_with_missing = df.columns[df.isnull().any()].tolist()
    print(f'Total Missing Values: {total_missing}')
    print(f'Rows with Missing Values: {rows_with_missing}')
    print(f'Columns with Missing Values: {", ".join(columns_with_missing)}')

    print('\nMissing Values Proportion:')
    missing_proportion = df.isnull().mean()
    plt.figure(figsize=(6, 3))
    sns.set(style="whitegrid")
    sns.barplot(x=missing_proportion.index, y=missing_proportion, color='red')
    plt.title('Proportion of Missing Values for Each Variable', fontsize=10, fontweight='bold')
    plt.xlabel('Variables', fontsize=8, fontweight='bold')
    plt.ylabel('Proportion of Missing Values', fontsize=8, fontweight='bold')
    plt.yticks(fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=8)    
    plt.grid(False)
    plt.show()

    # Duplicate rows
    print('\nDuplicate Rows:')
    num_duplicates = df.duplicated().sum()
    print(f'Number of Duplicate Rows: {num_duplicates}')
    print('\nMay the data guide you, and the Omnissiah bless your analysis.')