#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st


# In[3]:


from demo_streamlit import part2_streamlit, part3_streamlit, part1_streamlit, part4_streamlit


# In[ ]:


def main():
    
    liste_menu = ['Datasets downloaded from Kaggle', 'Dataset used for the Machine Learning part (grouped by providers + feature engineering)','Machine Learning on SMOTE', 'Machine Learning on RandomOverSampling']
    menu = st.sidebar.selectbox('Select the project presentation:', liste_menu)
    if menu == liste_menu[2]:
        part2_streamlit()
    elif menu == liste_menu[3]:
        part3_streamlit()
    elif menu == liste_menu[1]:
        part4_streamlit()
    else :
        part1_streamlit()
        
if __name__ == '__main__':
    main()
        

