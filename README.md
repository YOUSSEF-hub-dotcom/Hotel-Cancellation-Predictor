# 🏨 Hotel Reservation Cancellation Prediction  

This project focuses on predicting hotel reservation cancellations using **Machine Learning (XGBoost)**.  
The dataset contains hotel booking information such as reservation dates, guest details, and cancellation status.  

---

## 📌 Project Workflow
1. **Data Cleaning & Preprocessing**
   - Handled missing values with suitable techniques (mean, mode, 0, drop).
   - Removed outliers in ADR (average daily rate).
   - Converted categorical columns into numerical (One-Hot Encoding).

2. **Exploratory Data Analysis (EDA)**
   - Identified booking patterns (city vs resort hotels).
   - Checked cancellation rates across months, countries, and price ranges.
   - Visualized top features using plots and charts.

3. **Model Building**
   - Used **XGBoost Classifier** with GridSearchCV for hyperparameter tuning.
   - Best Parameters:
     ```
     {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.1, 
      'max_depth': 10, 'n_estimators': 200, 'subsample': 0.8}
     ```

4. **Results**
   - **Train Accuracy:** 86%  
   - **Test Accuracy:** 80%  
   - Precision/Recall balanced, with insights into the most important features.  

5. **Feature Importance**
   - Top features influencing cancellation:  
     - Previous Cancellations  
     - Special Requests  
     - Lead Time  
     - Arrival Month  

---

## 📊 Conclusion
- The model successfully predicts hotel reservation cancellations with ~80% accuracy.  
- Strong insights were drawn about customer booking behavior.  
- This project can help hotels reduce revenue loss by anticipating cancellations and adjusting strategies.

---

## 🚀 Tech Stack
- Python (Pandas, NumPy, Matplotlib, Seaborn, Plotly)
- Scikit-learn
- XGBoost
  linked in (www.linkedin.com/in/engineer-youssef-mahmoud-63b243361)
