# Social Media Usage and Emotional Well-Being

## Project Overview

This project focuses on developing a supervised machine-learning classification model to predict a user's **dominant emotional state** based on their social-media usage and engagement patterns.

The project covers the complete machine-learning lifecycle, including:

- Data loading and inspection
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Categorical variable encoding
- Feature scaling
- Model training and comparison
- Model selection and evaluation
- Model serialization
- Streamlit application development
- GitHub integration
- Streamlit Community Cloud deployment

The final model is integrated into an interactive Streamlit web application that allows users to enter their social-media usage characteristics and obtain a predicted emotional state.

---

## Problem Statement

Social-media usage can be associated with different patterns of emotional well-being. The objective of this project is to use structured demographic, usage and engagement data to develop a machine-learning classification model capable of predicting a user's dominant emotional state.

The target variable contains multiple emotional-state categories, including:

- Happiness
- Sadness
- Anger
- Anxiety
- Boredom
- Neutral

This makes the problem a **multiclass classification task**.

---

## Dataset

The project uses the **Social Media Usage and Emotional Well-Being** dataset obtained from Kaggle.

The dataset contains demographic information together with social-media usage and engagement variables.

### Input Features

| Feature | Type | Description |
|---|---|---|
| Age | Numerical | User's age |
| Gender | Categorical | User's gender |
| Platform | Categorical | Social-media platform used |
| Daily_Usage_Time | Numerical | Daily social-media usage time |
| Posts_Per_Day | Numerical | Number of posts per day |
| Likes_Received_Per_Day | Numerical | Likes received per day |
| Comments_Received_Per_Day | Numerical | Comments received per day |
| Messages_Sent_Per_Day | Numerical | Messages sent per day |

### Target Variable

**Dominant Emotional State**

Possible classes include:

- Happiness
- Sadness
- Anger
- Anxiety
- Boredom
- Neutral

---

## Project Workflow

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Preparation
     ↓
Categorical Encoding
     ↓
Train / Validation / Test Split
     ↓
StandardScaler
     ↓
Model Training
     ↓
Model Comparison
     ↓
Model Selection
     ↓
Final Model Evaluation
     ↓
Model & Preprocessing Serialization
     ↓
Streamlit Application
     ↓
GitHub
     ↓
Streamlit Community Cloud
