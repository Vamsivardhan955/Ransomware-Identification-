import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef)
import lime
from lime import lime_tabular
from imblearn.over_sampling import SMOTE
from collections import Counter

# Set page config
st.set_page_config(page_title="Ransomware Classifier", layout="wide")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Define IV/WoE function
def iv_woe(data, target, bins=10):
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns
    
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]})
        newDF = pd.concat([newDF, temp], axis=0)
    
    return newDF

# Main app
def main():
    st.title("Ransomware Classification Analysis")
    
    # File upload
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep='|')
    else:
        st.info("Please upload a CSV file using the sidebar")
        return
    
    # Dataset Exploration
    st.header("Dataset Exploration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("First 5 rows:")
        st.dataframe(df.head())
    
    with col2:
        st.write("Dataset shape:")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        st.write("Null values summary:")
        st.dataframe(df.isnull().sum().to_frame().T)
    
    # Class distribution
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    df.legitimate.value_counts().plot.pie(autopct='%.2f%%', labels=['Safe','Ransomware'], ax=ax)
    st.pyplot(fig)
    
    # Correlation Analysis
    st.header("Feature Correlation Analysis")
    
    numeric_df = df.drop(['Name','md5','legitimate'], axis=1, errors='ignore')
    corr_matrix = numeric_df.corr().abs()
    
    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(corr_matrix, ax=ax)
    st.pyplot(fig)
    
    # Feature Selection
    st.header("Feature Selection using IV/WoE")
    
    iv = iv_woe(df.drop(['Name','md5'], axis=1), 'legitimate')
    iv_sorted = iv.sort_values(by='IV', ascending=False)
    
    thresh = st.slider("IV Threshold", 0.0, 2.0, 1.0)
    features = iv_sorted[iv_sorted.IV > thresh]['Variable'].tolist()
    
    st.write(f"Selected Features ({len(features)}):")
    st.write(features)
    
    # Model Training
    st.header("Model Training")
    
    X = df[features]
    y = df['legitimate']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Handle class imbalance
    use_smote = st.checkbox("Apply SMOTE for class imbalance")
    if use_smote:
        smt = SMOTE()
        X_train, y_train = smt.fit_resample(X_train, y_train)
        st.write("Class distribution after SMOTE:", Counter(y_train))
    
    # Train model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluation
    st.subheader("Model Evaluation")
    
    pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    cm = confusion_matrix(y_test, pred)
    accuracy = (cm[1,1] + cm[0,0]) / cm.sum()
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)
    auc = roc_auc_score(y_test, pred_proba)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")
        st.metric("Precision", f"{precision:.2%}")
        st.metric("Recall", f"{recall:.2%}")
    
    with col2:
        st.metric("F1 Score", f"{f1:.2%}")
        st.metric("MCC", f"{mcc:.2f}")
        st.metric("AUC Score", f"{auc:.2f}")
    
    # Confusion Matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    st.pyplot(fig)
    
    # LIME Explanations
    st.header("Model Explainability with LIME")
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        verbose=False,
        mode='classification'
    )
    
    instance_idx = st.number_input("Enter test instance index", 0, len(X_test)-1, 0)
    
    if st.button("Generate Explanation"):
        exp = explainer.explain_instance(
            X_test.iloc[instance_idx].values,
            rf.predict_proba,
            num_features=5
        )
        
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)
        
        st.write("Explanation details:")
        st.write(exp.as_list())

if __name__ == "__main__":
    main()