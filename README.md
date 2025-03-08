# Student Performance Analysis: Predicting GPA & Academic Success

## Introduction

In this project I explore key factors influencing high school student performance. Using a dataset containing **2,392 student records**, I applied **regression models, classification techniques, and statistical tests** to analyze GPA determinants and predict academic success.

I welcome feedback or suggestions for further refinement!

## Dataset

The dataset was sourced from Kaggle:  
[**Students Performance Dataset**](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset/data)

It contains **various attributes** for each student:
- **Demographics**: Gender, Ethnicity, Age.
- **Academic Factors**: GPA (0.0 - 4.0 scale), Study Time, Absences, Grade Classification.
- **Parental Involvement**: Parental Education, Parental Support.
- **Extracurricular Activities**: Sports, Music, Volunteering, Other Activities.
- **Tutoring**: Whether the student receives academic tutoring.

## Objectives

1. **Identify key factors affecting student GPA**: Understand how study habits, parental support, and extracurriculars impact performance.
2. **Develop predictive models**: Train regression and classification models to forecast GPA and predict whether a student achieves a grade of **"C or more"**.
3. **Provide insights for educators and policymakers**: Help schools and organizations optimize interventions for student success.

## Methodology

### **Exploratory Data Analysis (EDA)**
- **Data Cleaning & Preprocessing**: Recoded categorical variables, handled missing values, and validated dataset integrity.
- **Descriptive Statistics**: Analyzed student demographics, GPA distributions, and key academic variables.
- **Visualization**: Used `ggplot2` to explore relationships between academic performance and external factors.

### **Statistical Testing**
- **T-Tests & ANOVA**: Assessed differences in GPA across **gender, sports participation, extracurriculars, and parental education levels**.
- **Correlation Analysis**: Found that **GPA is highly negatively correlated with absences (r = -0.92)** and slightly positively correlated with study time (r = 0.18).
- **Normality Testing**: Confirmed that GPA distribution does not follow normality (Jarque-Bera Test).

### **Regression Analysis**
- **Ordinary Least Squares (OLS) Regression**: Built a model predicting GPA with an **adjusted R² of 95.4%**.
- **Stepwise Regression (AIC)**: Refined the model by selecting the most significant variables.
- **Quantile Regression**: Explored predictor effects across different GPA levels.
- **Wald Test**: Verified that the OLS model performs well across quantiles.

### **Classification Models**
1. **Logistic Regression**: Predicted whether a student achieves **"C or more"** using key predictors.
   - Achieved **95.1% accuracy** on test data.
   - **ROC AUC: 0.9887**, showing excellent discriminatory power.
   - Fine-tuned **decision threshold** using Youden’s J statistic (best threshold: 0.41).
2. **Naïve Bayes Classifier**: Provided an alternative probabilistic model for predicting academic performance.
   - Achieved **92.3% accuracy**, slightly lower than logistic regression.
   - **ROC AUC: 0.9771**, confirming strong predictive performance.

## Key Findings

- **Absences are the strongest negative predictor of GPA**: More missed classes significantly lower student performance.
- **Study time positively influences GPA**, but the effect is relatively small compared to attendance.
- **Extracurricular activities (sports & music) correlate with higher GPA**.
- **Tutoring is a critical factor for academic success**: Students who receive tutoring have significantly higher GPAs.
- **Parental support has a significant impact**: Higher support levels are strongly associated with better grades.

## Recommendations

1. **Enhance Attendance Policies**: Schools should implement stricter attendance tracking to reduce absenteeism.
2. **Expand Tutoring Services**: Providing **accessible tutoring** can **significantly improve student performance**.
3. **Promote Extracurricular Engagement**: Encouraging **sports and music participation** benefits both academic and personal development.
4. **Parental Involvement Programs**: Schools should offer **parent workshops** to increase engagement and support for students.

## Limitations & Future Work

- **Dataset Bias**: The dataset may not account for **socioeconomic status, mental health, or school environment**.
- **Feature Expansion**: Future research could include **teacher effectiveness, classroom size, and curriculum difficulty**.
- **Machine Learning Enhancements**: Exploring **random forest, SVM, or deep learning models** could improve predictive accuracy.

## How to Run the Code

1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
