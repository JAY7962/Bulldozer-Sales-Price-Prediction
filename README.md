# 🚜 Bulldozer Sales Price Prediction

## 🚀 Project Overview

This project aims to predict the future sale price of bulldozers based on various features such as their configuration, usage metrics, sales history, and market conditions. This is a classic **regression problem** in machine learning, and we utilize the Scikit-learn implementation of the **Random Forest Regressor** as our primary model.

The dataset, sourced from a Kaggle competition, contains numerous features, including timestamps and categorical data with missing values. The notebook details the entire process from initial data exploration and cleaning to feature engineering, model training, hyperparameter tuning, and final prediction generation.

## 💾 Data

The project uses the `TrainAndValid.csv` file, which is assumed to be located in a specific directory (`"Downloads/bluebook-for-bulldozers/bluebook-for-bulldozers/TrainAndValid.csv"`).

* **Dataset Size:** The training and validation dataset contains over 400,000 entries and 53 features.
* **Target Variable:** The `SalePrice` is the target variable to be predicted.
* **Evaluation Metric:** The primary evaluation metric is the **Root Mean Squared Logarithmic Error (RMSLE)**, which is common for price prediction problems where relative error matters more than absolute error.

## 💻 Methodology

The project follows a standard machine learning workflow, with a particular focus on preparing time-series and mixed-type data for a forest-based model.

### 1. Data Exploration and Preprocessing

* **Loading and Time-Series Analysis:** The `saledate` column was loaded as a datetime object, and the data was sorted by date to respect the time-series nature of the problem.
    * Initial scatter plots of `SalePrice` vs. `saledate` and a histogram of `SalePrice` were used for visualization.
* **Feature Engineering (Date):** New time-series features were extracted from `saledate` (e.g., `saleYear`, `saleMonth`, `saleDay`, `saleDayOfWeek`, `saleDayOfYear`).
* **Handling Categorical Data:** All `object` type columns were converted to the pandas `category` data type to make them suitable for numeric encoding and to improve memory efficiency.
* **Numeric Encoding:** Categorical features were encoded into numerical representations using `pd.Categorical(content).codes + 1`. The `+1` handles missing values (which are typically encoded as `-1`).
* **Handling Missing Values (Imputation):**
    * **Numeric Features:** Missing values in numeric columns (`auctioneerID`, `MachineHoursCurrentMeter`) were imputed with the **median** of the respective column. A corresponding binary feature (`*_is_missing`) was created for each imputed column to retain information about the missingness.
    * **Categorical Features:** Missing categorical values were handled during the numeric encoding step (`+1` offset).

### 2. Model Training and Evaluation

* **Model Selection:** `RandomForestRegressor` was chosen, as it is a strong baseline for structured data and inherently handles categorical features (after they are numerically encoded) and non-linear relationships well.
* **Train/Validation Split:** The data was split temporally:
    * **Training Set (`X_train`, `y_train`):** Data up to and including the year 2011.
    * **Validation Set (`X_valid`, `y_valid`):** Data after the year 2011 (for simulating future predictions).
* **Baseline Model (Subset Training):** An initial model was trained on a small subset of the data (`max_samples=10000`) to quickly establish a performance baseline before full training.
* **Hyperparameter Tuning (Randomized Search):** `RandomizedSearchCV` was used to explore the hyperparameter space efficiently and find a better-performing model configuration.
* **Final Model Training:** An `ideal_model` was trained using the best parameters found (or close to them, with no `max_samples` constraint) on the full training dataset for the best possible performance.

## 📈 Results

The performance of the models was evaluated using the **Mean Absolute Error (MAE)** and the **Root Mean Squared Logarithmic Error (RMSLE)**.

| Metric | Baseline Model (`max_samples=10000`) | Ideal Model (Full Data) |
| :--- | :--- | :--- |
| **Train MAE** | 5466.24 | **2854.99** |
| **Valid MAE** | 7260.29 | **5985.95** |
| **Train RMSLE** | 0.255 | **0.141** |
| **Valid RMSLE** | 0.299 | **0.248** |
| **Valid $R^2$** | 0.830 | **0.882** |

The final "Ideal Model" significantly reduced the RMSLE on the validation set to **0.248**, demonstrating strong performance and a marked improvement over the subset model.

## 📊 Feature Importance

The `ideal_model` provides feature importances, indicating which features were most influential in predicting the `SalePrice`. The top 20 most important features are visualized in the notebook:

| Rank | Feature Name | Importance |
| :--- | :--- | :--- |
| 1 | `YearMade` | ~19.14% |
| 2 | `ProductGroupDesc` | ~9.60% |
| 3 | `Coupler_System_is_missing` | ~6.90% |
| 4 | `fiProductClassDesc` | ~6.40% |
| 5 | `fiBaseModel` | ~4.91% |
| ... | ... | ... |

**Key Takeaways:**
1.  **`YearMade`** (The age of the bulldozer) is overwhelmingly the most important predictor of its sale price.
2.  High-level descriptive features like **`ProductGroupDesc`** and **`fiProductClassDesc`** are also critical.

## 🔮 Making Predictions

The final model (`ideal_model`) was used to make predictions on the provided separate test dataset (`Train.csv` was re-read and preprocessed for this step).

1.  A `preprocess_data` function was created to ensure the test data underwent the *exact* same transformations as the training data, addressing issues like missing columns.
2.  Predictions were generated and saved to a new CSV file.

The resulting predictions were saved in a DataFrame with `SalesID` and `SalesPrice` columns, and then exported to: `Downloads/bluebook-for-bulldozers/bluebook-for-bulldozers/Test_predictions.csv`.

## ⚙️ How to Run the Notebook

### Prerequisites

You will need Python and the following libraries installed:

* `pandas`
* `numpy`
* `matplotlib`
* `scikit-learn`
