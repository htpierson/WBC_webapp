import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wbc = pd.read_csv('data.csv')

wbc_new = df_wbc[['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 
'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',  
'fractal_dimension_worst']]

wbc_clean = wbc_new.dropna()

wbc_col = wbc_clean.drop(columns = ['id', 'diagnosis'])

st.header('Wisconsin Breast Cancer Dataset')

st.write('''How do the columns correlate to each other?
''')

fig1 = plt.figure(figsize=(25, 25))
sns.heatmap(wbc_col.corr(), annot = True, cmap = 'gist_heat_r')
st.pyplot(fig1)



st.write('''How do characteristics of benign and metastatic tumors relate to each other?
''')

choice1 = st.multiselect(
    'What are characteristics would you like to observe on your x axis?',
    wbc_col.columns.values)

choice2 = st.multiselect(
    'What are characteristics would you like to observe on your y axis?',
    wbc_col.columns.values)

fig2 = sns.pairplot(wbc_clean, hue = 'diagnosis', palette= 'gist_heat',
            x_vars=choice1,
            y_vars=choice2)
st.pyplot(fig2)

st.write('You selected:', choice1, 'for your x axis')
st.write('You selected:', choice2, 'for your y axis')



st.write('''What is the distribution and correlation between two characteristics of benign and metastatic tumors when compared?
''')

choice3 = st.selectbox(
    "What characteristic would you like as your x axis?",
    wbc_col.columns.values)

choice4 = st.selectbox(
    "What characteristic would you like as your y axis?",
    wbc_col.columns.values)

fig3 = sns.jointplot(data = wbc_clean, x = choice3, y = choice4, hue = 'diagnosis',
              palette = 'gist_heat')
st.pyplot(fig3)

st.write('You selected', choice3, 'as your x axis')
st.write('You selected', choice4, 'as your y axis')



st.write('''How do benign and metastatic tumors relate to each other within one of their characteristics?
''')

choice5 = st.selectbox(
    "What characteristic would you like to observe?",
    wbc_col.columns.values)

fig4 = plt.figure(figsize=(12, 6))
sns.boxplot(data = wbc_clean, x = choice5, y = "diagnosis", palette = 'gist_rainbow_r')
st.pyplot(fig4)

st.write('You selected', choice3)
