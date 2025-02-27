# Machine Learning Model Comparison for Classification Task

This repository contains the implementation and comparison of various machine learning models for a classification task. The models include Random Forest, XGBoost, LightGBM, and a Neural Network. The performance of each model is evaluated using metrics such as Accuracy, Precision, Recall, and F1 Score. Additionally, hyperparameter tuning was performed for the Random Forest model to optimize its performance.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Models](#models)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Results](#results)
7. [Usage](#usage)
8. [Dependencies](#dependencies)
9. [License](#license)

---

## Project Overview
The goal of this project is to compare the performance of different machine learning models on a classification task. The models are trained and evaluated on a preprocessed dataset, and their performance metrics are recorded for comparison. Hyperparameter tuning was performed for the Random Forest model to further improve its performance.

---

## Dataset
The dataset used in this project is included in this repository.
---

## Preprocessing
The following preprocessing steps were applied to the dataset:
1. **Data Cleaning**: Handling missing values, removing duplicates, and correcting inconsistencies.
2. **Feature Engineering**: Creating new features or transforming existing ones to improve model performance.
3. **Scaling/Normalization**: Scaling numerical features to ensure uniformity.
4. **Encoding Categorical Variables**: Converting categorical variables into numerical formats using techniques like one-hot encoding or label encoding.
5. **Train-Test Split**: Splitting the dataset into training and testing sets (e.g., 80-20 split).

---

## Models
The following machine learning models were implemented and evaluated:
1. **Random Forest**
   - Accuracy: 0.9808
   - Precision: 0.8903
   - Recall: 0.6540
   - F1 Score: 0.7541
2. **XGBoost**
   - Accuracy: 0.9752
   - Precision: 0.7317
   - Recall: 0.7109
   - F1 Score: 0.7212
3. **LightGBM**
   - Accuracy: 0.9782
   - Precision: 0.7685
   - Recall: 0.7393
   - F1 Score: 0.7536
4. **Neural Network**
   - The neural network was trained for 50 epochs, and the best validation accuracy achieved was **0.9720**.

---

## Hyperparameter Tuning
Hyperparameter tuning was performed for the Random Forest model using `RandomizedSearchCV`. The goal was to optimize the model's performance by searching for the best combination of hyperparameters.

### Hyperparameter Search Space
The following hyperparameters were tuned:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, 30]
- `min_samples_split`: [2, 5]
- `min_samples_leaf`: [1, 2]
- `bootstrap`: [True]

### Tuning Process
- **Method**: Randomized Search with 10 iterations and 3-fold cross-validation.
- **Scoring Metric**: F1 Score.

### Results
- **Best Parameters**:
  ```python
  {'n_estimators': 300, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 30, 'bootstrap': True}
  ```
- **Best CV Score**: 0.9757
- **Test Set Performance**:
  - Accuracy: 0.9776
  - Precision: 0.7624
  - Recall: 0.7299
  - F1 Score: 0.7458

---

## Results
The performance metrics for each model are summarized below:

| Model          | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| Random Forest  | 0.9808   | 0.8903    | 0.6540 | 0.7541   |
| XGBoost        | 0.9752   | 0.7317    | 0.7109 | 0.7212   |
| LightGBM       | 0.9782   | 0.7685    | 0.7393 | 0.7536   |
| Neural Network | 0.9720   | -         | -      | -        |
| **Tuned Random Forest** | 0.9776 | 0.7624 | 0.7299 | 0.7458 |

---

## Usage
To reproduce the results or use the code in this repository, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Dikshitharyana/Ecommerce-Fraud-Detection
   ```
2. Install the required dependencies (see [Dependencies](#dependencies)).
3. Run the preprocessing script to prepare the dataset.
4. Train and evaluate the models using the provided scripts.

---

## Dependencies
The following Python libraries are required to run the code:
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- TensorFlow/Keras (for the Neural Network)

You can install the dependencies using:
```bash
pip install numpy pandas scikit-learn xgboost lightgbm tensorflow
```

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
