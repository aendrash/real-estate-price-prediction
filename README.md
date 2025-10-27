# 🏡 Ahmedabad House Price Prediction

This project focuses on analyzing and predicting real estate prices in **Ahmedabad** using machine learning.  
It is structured in three progressive levels:

| Level | Objective |
|------|-----------|
| **Fresher** | Perform Exploratory Data Analysis (EDA) and visualize key insights |
| **Mid-Level** | Train a regression model (LightGBM/XGBoost) and interpret key features |
| **Senior / Challenge** | Build a modular end-to-end ML pipeline with automated preprocessing, training, and evaluation using MLflow/Prefect (optional extension) |

---

## 📌 Dataset
The dataset used is the **Ahmedabad House Price dataset from Kaggle**.  
It contains property-related features such as:
- Area
- Location
- Property type
- Number of rooms
- Price (Target Variable)

---

## 🧹 1. Data Cleaning & Preprocessing
Implemented inside: `preprocessing.py`

Steps performed:
- Remove duplicates & missing price entries
- Clean price formatting (`₹`, `Lac`, commas → numeric float)
- Clean and standardize area values
- Encode categorical features using **Label Encoding**
- Saves processed dataset into `data/processed/cleaned_data.csv`

```bash
raw_data.csv → clean_data() → encode_data() → cleaned_data.csv
```

---

## 🔍 2. Exploratory Data Analysis (EDA)
Notebook: `2. EDA.ipynb`

Key Insights:
- Relationship between area and price  
- Popular location clusters  
- Price distribution across property types  
- Outlier visualization  

> Example insight: Larger area properties in premium localities show significantly higher price variation than outskirts.

---

## 🤖 3. Model Training
Implemented inside: `train.py`  
Model used: **LightGBM Regressor**

Training includes:
- Train-test split (80/20)
- Log transformation on price for normalization
- Model evaluation using RMSE and R² Score

**Outputs:**
- Saved model → `data/processed/lgbm_model.pkl`

---

## ⚙️ 4. Full ML Pipeline
Script: `pipeline.py`

The pipeline performs:
1. Load → Clean → Encode Data  
2. Train LightGBM Model  
3. Predict & Evaluate Performance  

Run pipeline:

```bash
python pipeline.py
```

**Example Output:**

```yaml
🔹 Starting preprocessing...
🔹 Training model...
🔹 Evaluating...
Final Metrics: {'RMSE': <value>, 'R2': <value>}
```

---

## 🧠 Feature Importance (Mid-Level Requirement)
LightGBM provides feature importance to explain key contributors in price prediction.

Common influential features (expected):
- Property Location
- Area (`value_area`)
- Property Type
- Number of Bedrooms

Interpretation is shown inside `3. training ml models.ipynb`.

---

## 🏗️ Senior Challenge: Extend to MLOps (Optional)
To upgrade to a production-grade ML workflow, integrate:
- **Experiment tracking:** MLflow  
- **Orchestration:** Prefect  
- **CI/CD:** Automated retraining  
- **Model Registry:** Deploy best-performing models  

---

## 📁 Project Structure
```
.
├── data/
│   ├── raw/
│   ├── processed/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── pipeline.py
├── notebooks/
│   ├── 1. data extraction and data cleaning.ipynb
│   ├── 2. EDA.ipynb
│   ├── 3. training ml models.ipynb
```

---

## ✅ Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn lightgbm joblib
```

---

## 🎯 Summary
This project demonstrates:
- Real-world data cleaning  
- Insight extraction from EDA  
- Training a competitive gradient boosting model  
- Building a reproducible ML pipeline  

---

## ⭐ Contributions & Improvements Welcome!
```
Pull requests • Bug fixes • Performance tuning
```

---

**Author:** <Your Name>
