# Loan Data Analysis Project

This project is designed to analyze and predict loan approvals. It includes scripts for preprocessing data, conducting statistical analysis, and applying machine learning algorithms to classify if a loan will get approved.

## Project Structure

The project consists of the following files:

### SQL_loan_data.sql

- **Purpose**: This SQL script is used for initial data extraction, cleaning, and preparation. It queries the loan database to select relevant columns, handles missing values, and filters the dataset as required for the analysis.
- **Usage**: Execute this script in your SQL database management tool (such as PostgreSQL, MySQL, or Microsoft SQL Server) to prepare the data for analysis.

### loans_ANN_classifier.py

- **Purpose**: This Python script utilizes Artificial Neural Networks (ANNs) to classify loan applications as likely to default or not. It includes data preprocessing, training the ANN model, and evaluating its performance using accuracy and loss metrics.
- **Usage**: Run this script with Python 3.x. Ensure you have necessary libraries installed (`numpy`, `pandas`, `tensorflow`, `keras`) by running `pip install numpy pandas tensorflow keras`.

### loan_data_regression.R

- **Purpose**: This R script applies regression analysis techniques to predict the probability of default based on various loan applicant features. It includes data visualization, model fitting, and result interpretation.
- **Usage**: Open and run this script in RStudio or any other R environment. Make sure all required packages (`ggplot2`, `dplyr`, `caret`) are installed using `install.packages("package_name")`.

### train.csv
- **Purpose**: This file holds the labeled dataset to train the algorithms.
- **Usage**: Import this file to train your models.

### test.csv
- **Purpose**: This file holds the unlabeled dataset to make predicitons.
- **Usage**: Import this file after you have trained your models to predict the loan success.


## Getting Started

To run this project, follow these steps:

1. **Prepare the Database**:
   - Run the `SQL_loan_data.sql` script in your database management system to so you can have a look at the data.
   
2. **Classification with ANN**:
   - Execute the `loans_ANN_classifier.py` in a Python environment. This script will train the ANN model and output the classification accuracy and loss. It will also make predictions for the data in test.csv

3. **Regression Analysis**:
   - Load and execute the `loan_data_regression.R` script in RStudio to perform regression analysis.

## Requirements

- SQL Database Management System (DBMS)
- Python 3.x
- Libraries: `numpy`, `pandas`, `tensorflow`, `keras`
- R and RStudio
- R packages: `ggplot2`, `dplyr`, `caret`

## Contributing

If you would like to contribute to this project or suggest improvements, please fork the repository and submit a pull request or open an issue with your suggestions.

## License

This project is free to use without any restrictions.
