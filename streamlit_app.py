import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Title of the app
st.title('🌸 Iris Species Prediction App')
st.info('This app builds a machine learning model to predict Iris species!')

# Load dataset
data_path = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

try:
    df = pd.read_csv(data_path)  # Load the dataset from the specified path
except FileNotFoundError:
    st.error("File not found. Please check the file path.")
    st.stop()

# Rename columns for easier access
df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']
species_names = list(df['species'].unique())

# Display raw data
with st.expander('Data'):
    st.write('**Raw data**')
    st.dataframe(df)

    st.write('**X** (Features)')
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.write('**y** (Target)')
    y_raw = df['species']
    st.dataframe(y_raw)

# Data visualization
with st.expander('Data visualization'):
    # Ensure species is a string for color coding
    df['species'] = df['species'].astype(str)
    st.scatter_chart(data=df, x='sepal length (cm)', y='sepal width (cm)', color='species')

# Sidebar for input features
with st.sidebar:
    st.header('Input features')
    sepal_length = st.slider('Sepal length (cm)', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.slider('Sepal width (cm)', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.slider('Petal length (cm)', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.slider('Petal width (cm)', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

    # Create DataFrame for input features
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=df.columns[:-1])

with st.expander('Input features'):
    st.write('**Input flower**')
    st.dataframe(input_df)

# Encode target variable
target_mapper = {species: i for i, species in enumerate(species_names)}
df['species'] = df['species'].map(target_mapper)
y_raw = df['species']

# Model training
clf = RandomForestClassifier()
clf.fit(X_raw, y_raw)

# Prediction
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

# Display predicted species
st.subheader('Predicted Species Probability')
df_prediction_proba = pd.DataFrame(prediction_proba, columns=species_names)
st.dataframe(df_prediction_proba)

predicted_species = [k for k, v in target_mapper.items() if v == prediction[0]][0]
st.success(f'Predicted species: {predicted_species}')
