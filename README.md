# Soil Temperature Modeling Using Ensemble Machine Learning

## Overview
Soil temperature significantly influences plant development, nutrient cycling, and microbial processes. This project models soil temperature in Grand Forks, ND using environmental data and interpretable ensemble machine learning models. 

We compare three validation strategies—Train/Test Split, K-Fold Cross-Validation, and Repeated K-Fold Cross-Validation—to assess predictive reliability and generalization.

Unlike prior deep learning approaches, this work focuses on lightweight, interpretable models suitable for real-world agricultural and environmental applications.

## Methods
### Data
- Data was collected from the Grand Forks station of the North Dakota Agricultural Weather Network (NDAWN)
- Environmental features (e.g., air temperature, precipitation, solar radiation).
- Soil temperature measurements at multiple depths: 10 cm, 50 cm, and 100 cm.

### Preprocessing
- Missing value imputation.
- Feature scaling (standardization).

### Validation Strategies
- Train/Test Split: Simple partition into training and test sets.
- K-Fold Cross-Validation: Dataset split into k folds; each fold serves as test data once.
- Repeated K-Fold: K-Fold repeated multiple times to reduce variance in evaluation.

### Models
- XGBoost
- Gradient Boosting
- Random Forest
- AdaBoost

### Evaluation Metrics
- Runtime
- RMSE (Root Mean Squared Error)
- Error Variance

## Key Findings
- Train/Test Split: Fastest, but prone to variability and often underestimates performance.
- Repeated K-Fold: Most stable, but computationally expensive.
- K-Fold CV: Balanced choice with strong performance and reasonable speed.

Among models, XGBoost and Gradient Boosting provided the most accurate, compact predictions across depths and seasonal conditions.
Feature importance varies by season and depth, highlighting the value of context-aware modeling.

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

## Contact information for questions and concerns

- Name: Tyrike Daniels  
- Email: tyriketheboss@gmail.com  
- GitHub Profile: [github link](https://github.com/tyrikedaniels)
- LinkedIn Profile: [linkedin link](www.linkedin.com/in/tyrike-daniels-255b84275)

