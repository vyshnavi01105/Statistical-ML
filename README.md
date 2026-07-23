# 🩺 Diabetes Prediction - Exploratory Data Analysis (EDA)

## 📌 Project Overview

This project performs **Exploratory Data Analysis (EDA)** on the **Pima Indians Diabetes Dataset**. The objective is to understand the dataset, identify patterns, detect missing or inconsistent values, analyze relationships between features, and prepare the data for future machine learning models.

---

## 📂 Dataset

- **Dataset:** Pima Indians Diabetes Dataset
- **Records:** 768
- **Features:** 9
- **Target Variable:** `Outcome`
  - **0** → Non-Diabetic
  - **1** → Diabetic

### Features

| Feature | Description |
|----------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic likelihood of diabetes |
| Age | Age of the patient |
| Outcome | Diabetes status |

---

## 🎯 Objectives

- Perform statistical analysis of the dataset.
- Identify missing and duplicate values.
- Detect outliers.
- Analyze feature distributions.
- Understand correlations among variables.
- Generate meaningful insights through visualizations.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 📊 Exploratory Data Analysis

The notebook includes:

- Dataset Overview
- Shape of Dataset
- Data Types
- Missing Value Analysis
- Duplicate Value Check
- Descriptive Statistics
- Mean
- Median
- Mode
- Variance
- Standard Deviation
- Minimum & Maximum Values
- Quartiles
- Range
- Skewness
- Kurtosis
- Covariance Matrix
- Correlation Matrix
- Correlation Heatmap
- Histograms
- Boxplots
- Pair Plot
- Scatter Plot
- Count Plot
- Outlier Detection (IQR Method)
- Zero Value Analysis
- Summary Statistics

---

## 📈 Key Insights

- The dataset contains **768 patient records** with **9 numerical features**.
- No NULL values are present.
- Several medical features contain unrealistic zero values that should be treated as missing values.
- The dataset is moderately imbalanced with more non-diabetic than diabetic patients.
- Glucose shows the strongest positive correlation with diabetes.
- BMI, Age, Pregnancies, and Diabetes Pedigree Function also contribute significantly to diabetes prediction.
- Outliers are present in several numerical features and should be handled appropriately before model training.

---

## 📁 Project Structure

```
Diabetes-EDA/
│
├── diabetes.csv
├── DIABETES.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

### Clone the repository

```bash
git clone https://github.com/your-username/Diabetes-EDA.git
```

### Navigate to the project

```bash
cd Diabetes-EDA
```

### Install required libraries

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open **DIABETES.ipynb** and run all cells.

---

## 📷 Visualizations

The project includes:

- Correlation Heatmap
- Histograms
- Boxplots
- Pair Plot
- Scatter Plot
- Outcome Count Plot

These visualizations help understand the distribution of variables, detect outliers, and identify relationships between features.

---

## 📌 Conclusion

The exploratory analysis provides valuable insights into the diabetes dataset and highlights important features that influence diabetes prediction. The dataset is suitable for predictive modeling after preprocessing steps such as handling zero values, treating outliers, and feature scaling.

---

## 👩‍💻 Author

**Vyshnavi**

---

## ⭐ If you found this project useful, don't forget to star the repository!
