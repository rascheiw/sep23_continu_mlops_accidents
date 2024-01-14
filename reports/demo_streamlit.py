#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.metrics import classification_report_imbalanced
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, f_regression, mutual_info_regression, RFE, RFECV
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


def part1_streamlit():
    test = pd.read_csv('Test-1542969243754.csv')
    test_benef = pd.read_csv('Test_Beneficiarydata-1542969243754.csv')
    test_in = pd.read_csv('Test_Inpatientdata-1542969243754.csv')
    test_out = pd.read_csv('Test_Outpatientdata-1542969243754.csv')
    train = pd.read_csv('Train-1542865627584.csv')
    train_benef = pd.read_csv('Train_Beneficiarydata-1542865627584.csv')
    train_in = pd.read_csv('Train_Inpatientdata-1542865627584.csv')
    train_out = pd.read_csv('Train_Outpatientdata-1542865627584.csv')
    st.title("Data Exploration of the datasets downloaded from Kaggle")
    st.image('hcp1.png')
    st.markdown('[Click here to access the project on Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis?resource=download) ')
    df = st.selectbox( "Choose the dataset to display:", ['test', 'test_benef', 'test_in', 'test_out',  'train', 'train_benef', 'train_in', 'train_out'] )
    if df == 'test':
        line_to_plot = st.slider("select le number of lines to show", min_value=5, max_value=20)
        st.dataframe(test.head(line_to_plot))
    elif df == 'test_benef':
        line_to_plot = st.slider("select le number of lines to show", min_value=5, max_value=20)
        st.dataframe(test_benef.head(line_to_plot))
    elif df == 'test_in':
        line_to_plot = st.slider("select le number of lines to show", min_value=5, max_value=20)
        st.dataframe(test_in.head(line_to_plot))
    elif df == 'test_out':
        line_to_plot = st.slider("select le number of lines to show", min_value=5, max_value=20)
        st.dataframe(test_out.head(line_to_plot))
    elif df == 'train':
        line_to_plot = st.slider("select le number of lines to show", min_value=5, max_value=20)
        st.dataframe(train.head(line_to_plot))
    elif df == 'train_benef':
        line_to_plot = st.slider("select le number of lines to show", min_value=5, max_value=20)
        st.dataframe(train_benef.head(line_to_plot))
    elif df == 'train_in':
        line_to_plot = st.slider("select le number of lines to show", min_value=5, max_value=20)
        st.dataframe(train_in.head(line_to_plot))
    elif df == 'train_out':
        line_to_plot = st.slider("select le number of lines to show", min_value=5, max_value=20)
        st.dataframe(train_out.head(line_to_plot)) 
    return

def part4_streamlit():
    train_provider = pd.read_csv('train_provider.csv', index_col= 0)
    feats = train_provider.drop('PotentialFraud', axis=1)
    target = train_provider['PotentialFraud']
    st.title("Information regarding the dataset used for the Machine Learning part (grouped by providers + feature engineering)")
    st.image('hcp1.png')
    st.header("Features")
    line_to_plot1 = st.slider("select le number of lines to show", min_value=5, max_value=20)
    st.dataframe(feats.head(line_to_plot1))
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25)
    st.header( "Target distribution")
    st.write(target.value_counts(normalize = True))
    st.header( "Target distribution of the training dataset")
    st.write(y_train.value_counts(normalize = True))
    return



def part2_streamlit():
    train_provider = pd.read_csv('train_provider.csv', index_col= 0)
    feats = train_provider.drop('PotentialFraud', axis=1)
    target = train_provider['PotentialFraud']
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25)
    
    smo = SMOTE()
    X_sm, y_sm = smo.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    
    scaler.fit(X_sm)
    X_sm_scaler = scaler.transform(X_sm)
    X_test_sm_scaler = scaler.transform(X_test)
    pca = PCA(n_components = 0.90)
    pca.fit(X_sm_scaler)
    X_sm_scaler_pca = pca.fit_transform(X_sm_scaler)
    X_test_sm_scaler_pca = pca.transform(X_test_sm_scaler)

    
    
    st.title("Score of the different models with the scaled SMOTE dataset")
    st.image('hcp1.png')
    choose_a_model = st.selectbox('Select a model', ['Random Forest', 'KNN', 'Decision Tree', 'XGBoost'])
    if choose_a_model == 'Random Forest':
        st.write("The best paramters detected by the grid search were: {'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 1000}")
        nest = st.selectbox('Select the number of estimtors', [10, 50, 100, 250, 500, 1000])
        msl = st.selectbox('Select the min samples leaf', [1, 3, 5])
        mf = st.selectbox('Select the max features', ['sqrt', 'log2'])
        model = RandomForestClassifier(n_estimators = nest, min_samples_leaf = msl, max_features = mf)
        model.fit(X_sm_scaler_pca, y_sm)
        y_pred = model.predict(X_test_sm_scaler_pca)
        st.subheader("Classification Report")
        st.text(classification_report_imbalanced(y_test, y_pred))
        st.subheader("Score")
        st.text(model.score(X_test_sm_scaler_pca, y_test))
    if choose_a_model == 'KNN':
        st.write("The best paramters detected by the grid search were: {'metric': 'manhattan', 'n_neighbors': 2}")
        nn = st.selectbox('Select the number of neighbors', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        met = st.selectbox('Select the metric', ['minkowski', 'manhattan', 'chebyshev'])
        model = neighbors.KNeighborsClassifier(n_neighbors = nn, metric = met)
        model.fit(X_sm_scaler_pca, y_sm)
        y_pred = model.predict(X_test_sm_scaler_pca)
        st.subheader("Classification Report")
        st.text(classification_report_imbalanced(y_test, y_pred))
        st.subheader("Score")
        st.text(model.score(X_test_sm_scaler_pca, y_test))
    if choose_a_model == 'Decision Tree':
        st.write("The best paramters detected by the grid search were: {'criterion': 'entropy', 'max_depth': 12}")
        crit = st.selectbox('Select the criterion', ['gini', 'entropy'])
        maxd = st.selectbox('Select the mmaximum depth', [2,4,6,8,10,12])
        model = DecisionTreeClassifier(criterion = crit, max_depth = maxd)
        model.fit(X_sm_scaler_pca, y_sm)
        y_pred = model.predict(X_test_sm_scaler_pca)
        st.subheader("Classification Report")
        st.text(classification_report_imbalanced(y_test, y_pred))
        st.subheader("Score")
        st.text(model.score(X_test_sm_scaler_pca, y_test))
    if choose_a_model == 'XGBoost':
        boost = st.selectbox('Select the booster', ['gbtree', 'gblinear', 'dart'])
        obj = st.selectbox('Select the objective', ['binary:logistic', 'reg:squarederror'])
        lrate = st.slider('Select the learning_rate', min_value=0.00, max_value=1.00, step= 0.01)
        params = {'booster': boost, 'learning_rate': lrate, 'objective': obj}
        train = xgb.DMatrix(X_sm_scaler_pca, y_sm)
        test = xgb.DMatrix(X_test_sm_scaler_pca, y_test)
        xgb2 = xgb.train(params = params, dtrain = train, num_boost_round = 700, early_stopping_rounds = 15, evals = [(train, 'train'), (test, 'eval')])
        preds = xgb2.predict(test)
        xgbpreds = pd.Series(np.where(preds > 0.5, 1, 0))
        st.subheader("Classification Report")
        st.text(classification_report_imbalanced(y_test, xgbpreds))
    return

def part3_streamlit():
    train_provider = pd.read_csv('train_provider.csv', index_col= 0)
    feats = train_provider.drop('PotentialFraud', axis=1)
    target = train_provider['PotentialFraud']
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25)
#   if st.checkbox( "Display the target distribution") :
#       st.write(target.value_counts(normalize = True))
#   if st.checkbox( "Display the target distribution of the training dataset") :
#       st.write(y_train.value_counts(normalize = True))
    
    ros = RandomOverSampler()
    X_ro, y_ro = ros.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    
    scaler.fit(X_ro)
    X_ro_scaler = scaler.transform(X_ro)
    X_test_ro_scaler = scaler.transform(X_test)
    pca = PCA(n_components = 0.90)
    pca.fit(X_ro_scaler)
    X_ro_scaler_pca = pca.fit_transform(X_ro_scaler)
    X_test_ro_scaler_pca = pca.transform(X_test_ro_scaler)


    st.title("Score of the different models with the scaled RandomOverSampling dataset")
    st.image('hcp1.png')
    choose_a_model1 = st.selectbox('Select a model', ['Random Forest', 'KNN', 'Decision Tree', 'XGBoost'])
    if choose_a_model1 == 'Random Forest':
        st.write("The best paramters detected by the grid search were: {'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 100}")
        nest = st.selectbox('Select the number of estimtors', [10, 50, 100, 250, 500, 1000])
        msl = st.selectbox('Select the min samples leaf', [1, 3, 5])
        mf = st.selectbox('Select the max features', ['sqrt', 'log2'])
        model = RandomForestClassifier(n_estimators = nest, min_samples_leaf = msl, max_features = mf)
        model.fit(X_ro_scaler_pca, y_ro)
        y_pred = model.predict(X_test_ro_scaler_pca)
        st.subheader("Classification Report")
        st.text(classification_report_imbalanced(y_test, y_pred))
        st.subheader("Score")
        st.text(model.score(X_test_ro_scaler_pca, y_test))
    if choose_a_model1 == 'KNN':
        st.write("The best paramters detected by the grid search were: {'metric': 'manhattan', 'n_neighbors': 1}")
        nn = st.selectbox('Select the number of neighbors', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        met = st.selectbox('Select the metric', ['minkowski', 'manhattan', 'chebyshev'])
        model = neighbors.KNeighborsClassifier(n_neighbors = nn, metric = met)
        model.fit(X_ro_scaler_pca, y_ro)
        y_pred = model.predict(X_test_ro_scaler_pca)
        st.subheader("Classification Report")
        st.text(classification_report_imbalanced(y_test, y_pred))
        st.subheader("Score")
        st.text(model.score(X_test_ro_scaler_pca, y_test))
    if choose_a_model1 == 'Decision Tree':
        st.write("The best paramters detected by the grid search were: {'criterion': 'entropy', 'max_depth': 12}")
        crit = st.selectbox('Select the criterion', ['gini', 'entropy'])
        maxd = st.selectbox('Select the mmaximum depth', [2,4,6,8,10,12])
        model = DecisionTreeClassifier(criterion = crit, max_depth = maxd)
        model.fit(X_ro_scaler_pca, y_ro)
        y_pred = model.predict(X_test_ro_scaler_pca)
        st.subheader("Classification Report")
        st.text(classification_report_imbalanced(y_test, y_pred))
        st.subheader("Score")
        st.text(model.score(X_test_ro_scaler_pca, y_test))
    if choose_a_model1 == 'XGBoost':
        boost = st.selectbox('Select the booster', ['gbtree', 'gblinear', 'dart'])
        obj = st.selectbox('Select the objective', ['binary:logistic', 'reg:squarederror'])
        lrate = st.slider('Select the learning_rate', min_value=0.00, max_value=1.00, step= 0.01)
        params = {'booster': boost, 'learning_rate': lrate, 'objective': obj}
        train = xgb.DMatrix(X_ro_scaler_pca, y_ro)
        test = xgb.DMatrix(X_test_ro_scaler_pca, y_test)
        xgb2 = xgb.train(params = params, dtrain = train, num_boost_round = 700, early_stopping_rounds = 15, evals = [(train, 'train'), (test, 'eval')])
        preds = xgb2.predict(test)
        xgbpreds = pd.Series(np.where(preds > 0.5, 1, 0))
        st.subheader("Classification Report")
        st.text(classification_report_imbalanced(y_test, xgbpreds))
    return
    
