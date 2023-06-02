import streamlit as st
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import os 


import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

import pycaret
from pycaret.classification import setup,compare_models,pull,save_model,ClassificationExperiment
from pycaret.regression import setup,compare_models,pull,save_model,RegressionExperiment


st.title("Machine Learning App")

if os.path.exists("source.csv"):
    df = pd.read_csv("source.csv",index_col=None)
    

with st.sidebar:
    st.image("https://builtin.com/sites/www.builtin.com/files/styles/og/public/2021-12/machine-learning-examples-applications.png")
    st.header("Automated Machine Learning application")
    st.subheader("This Application Is Made For Learning Machine Model")
    st.caption("Choose Your Parameter Here To Work On The Application ")
    choose =st.radio("Choose your options ",["Dataset","Analysis","Training","Download"])
if choose=="Dataset":
    st.write("Please upload your dataset here. Only .csv files allowed")
    Available_Datasets=[filename for filename in os.listdir()if filename.endswith('.csv')]
    selected_Datasets=st.selectbox('Select Datasets',Available_Datasets)
    
if selected_Datasets:
    df=pd.read_csv(selected_Datasets,index_col=None)
    df.to_csv("sourcev.csv", index = None)
    st.dataframe(df)
    st.success('Dataset Suessfully Loaded')
 else:
    st.error('Error: No Dataset Avaialble')


        
if choose=="Analysis":
    st.subheader("Perform profiling on dataset")
    if st.sidebar.button("Do Analysis"):
        profile_report=df.profile_report()
        st_profile_report(profile_report)
    
    
if choose=="Training":    
    st.header("Start Training Your Model Now")
    choice =st.sidebar.selectbox("Select Your Techniques:",["Classification","Regression"])
    target = st.selectbox("Select You Target Variable ",df.columns)
    if choice == "Classification":
        if st.sidebar.button("Classification Train"):
            s1=ClassificationExperiment()
            s1.setup(data=df,target=target)
        
            setup_df=s1.pull()
            
            st.info("The setup data is as follows:")
            st.table(setup_df)
            
            best_model1=s1.compare_models()
            compare_model=s1.pull()
            st.info("The comparison  of models is as follows:")
            st.table(compare_model)
            best_model1
            s1.save_model(best_model1,"Machine Learning Model")
    else:
        if st.sidebar.button("Regression Train"):
            s2=RegressionExperiment()
            s2.setup(data=df,target=target)
            
            setup_df=s2.pull()
            
            st.info("The setup data is as follows:")
            st.table(setup_df)
                 
            best_model2=s2.compare_models()
            compare_model=s2.pull()
            st.info("The comparison  of models is as follows:")
            st.table(compare_model)
            best_model2
            s2.save_model(best_model2,"Machine Learning Model")
            
if choose=="Download":
    with open("Machine Learning Model.pkl","rb") as f:
        st.caption("Download your model from here:")
        st.download_button("download the file ", f ,"Machine Learning Model.pkl")
