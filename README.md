# Evaluating Validation Strategies for Soil Temperature Prediction with Machine Learning
Soil Temperature Modeling Using Ensemble Machine Learning

## Overview
Soil temperature critically affects plant growth, nutrient uptake, and microbial activity. This project models soil temperature in Grand Forks, ND using environmental data and interpretable machine learning models. We evaluate three validation techniques—Train/Test Split, K-Fold Cross-Validation, and Repeated K-Fold—to assess model reliability and accuracy.
Unlike prior deep learning work, this study emphasizes lightweight, interpretable models for practical agricultural and environmental applications.

## Methods
- Data: Environmental features and soil temperature measurements at different depths (e.g., 10cm, 50cm, 100cm).
- Preprocessing: Handling missing values and feature scaling.
- Validation Techniques:
  - Train/Test Split: Simple train-test division.
  - K-Fold CV: Data split into k folds; model trains on k-1 and tests on the remaining fold.
  - Repeated K-Fold: Multiple repetitions of K-Fold for stability.
- Models: Ensemble regressors including XGBoost, Gradient Boosting, Random Forest, and AdaBoost.
- Evaluation: Runtime, RMSE (root mean squared error), and error variance.

## Key Findings
Train/Test is fastest but often misses best performance and variance. Repeated K-Fold is slightly better but much slower. K-Fold strikes the best balance.
XGBoost and Gradient Boosting achieved compact, accurate predictions across depths and seasons.
Feature importance varies by season and depth, underscoring the need for tailored models.

## Usage
1. Clone the repository:
```
git clone https://github.com/YourUsername/soil-temp-modeling.git
cd soil-temp-modeling
```

2. Install required packages
```
pip install -r requirements.txt
```
<sub>Or manually:</sub>
```
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

3. Run the code
```
python src/main_model.py
python src/main_val.py
```

Output files will be saved in the results/ directory.

## Requirements
Python 3.8+

Packages:
```
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
```

## File Structure

```
.
├── requirements.txt
├── README.md
├── data/
│   └── Grand Forks_daily updated.csv
├── results/
│   ├── eval_val.csv
│   ├── eval_val.png
│   ├── model_summary_table.csv
│   └── validation_metrics.csv
└── src/
    ├── main_model.py
    ├── main_val.py
    └── utils/
        ├── model.py
        ├── plot.py
        ├── preprocess.py
        └── val.py
```

## References

Ahmasebi Nasab, M., Pattanayak, S., Williams, T. W., Sharifan, A., Raheem, Y., & Fournier, C. (2024). Daily soil temperature simulation at different depths in the Red River Basin: A long short-term memory approach. Modeling Earth Systems and Environment, 1–12.
James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An introduction to statistical learning: With applications in R (Vol. 103). Springer.
BioRender. (n.d.). BioRender. Retrieved July 21, 2025, from https://www.biorender.com
Redregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (n.d.). sklearn.model_selection.GridSearchCV. scikit-learn. Retrieved July 21, 2025, from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 

## Contact / Questions

- Tyrike Daniels  
- 📧 tyriketheboss@gmail.com  
- 🌐 [GitHub Profile](https://github.com/tyrikedaniels)

