# Energy Consumption Prediction

This project predicts household energy consumption using machine learning models.  
Dataset: **energydata_complete.csv**

## 🚀 Steps in the Project
1. **Install Libraries**  
   `pip install lightgbm xgboost scikit-learn matplotlib seaborn pandas numpy joblib`

2. **Data Preprocessing**
   - Convert `date` column to `hour` & `day`.
   - Handle missing values.
   - Split features (`X`) and target (`y` = Appliances).
   - Train-test split and standardization.

3. **Models Used**
   - Linear Regression
   - Random Forest
   - XGBoost
   - LightGBM

4. **Evaluation Metrics**
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² Score

5. **Results**
   | Model              | RMSE  | MAE  | R²   |
   |--------------------|-------|------|------|
   | Linear Regression  | 91.14 | 52.59| 0.17 |
   | Random Forest      | 67.81 | 32.23| 0.54 |
   | XGBoost            | 70.55 | 35.69| 0.50 |
   | LightGBM           | 69.93 | 35.38| 0.51 |

   🏆 **Best Model: Random Forest**

6. **Outputs**
   - Saved best model → `best_energy_model.pkl`
   - Prediction vs Actual plot
   - Feature importance plot

## ▶️ How to Run
```bash
python main.py
```

## 📊 Visualizations
- RMSE comparison barplot
- Prediction vs Actual scatterplot
- Feature importance (tree-based models)

---
