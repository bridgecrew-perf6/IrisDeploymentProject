#!/usr/bin/env python
# coding: utf-8
# Author: Meetu
# In[ ]:


import streamlit as st
import joblib
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Flower Prediction App", page_icon=None, layout='centered', initial_sidebar_state='auto')

# In[ ]:


@st.cache(allow_output_mutation=True)
def load(model_path):
    flower = joblib.load(model_path)
    return flower


# In[ ]:


def inference(row, model, cols):
    df = pd.DataFrame([row], columns = cols)
    features = pd.DataFrame([row], columns = cols)
    if (model.predict(features)=='Iris-virginica'):
        return "This is Iris-virginica!"
    elif(model.predict(features)=='Iris-versicolor'):
        return "This is Iris-versicolor!"
    else: 
        return "This is Iris-setosa!"


# In[ ]:


st.title('Flower Prediction App')
st.write('The iris dataset contains three classes of flowers, Versicolor, Setosa, Virginica, and each class contains 4 features, ‘Sepal length’, ‘Sepal width’, ‘Petal length’, ‘Petal width’. The aim of the iris flower classification is to predict flowers based on their specific features.')
image = Image.open('C:/Users/meetu/Downloads/myDSPortfolio/deployment/irisDeployment/data/flower.jfif')
st.image(image, use_column_width=True)
st.write('Please fill in the details in the left sidebar and click on the button below!')

SepalLength =           st.sidebar.number_input("Sepal Length", 0.1, 10.0, 0.5) # label, min_value, max_vale, step(optional)
SepalWidth =   st.sidebar.number_input("Sepal Width", 0.1, 10.0, 0.5)
PetalLength =       st.sidebar.slider("Petal Length", 0.1, 10.0, 0.5)
PetalWidth = st.sidebar.slider("Petal Width", 0.1, 10.0, 0.5)

row = [SepalLength, SepalWidth, PetalLength, PetalWidth]


# In[ ]:


if (st.button('Find Iris flower category')):
    cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    model = load('models/flower.jl')
    result = inference(row, model, cols)
    st.write(result)
