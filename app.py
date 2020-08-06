import pandas as pd
import numpy as np
import streamlit as st
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib 
import pickle




# Attribute Information:

# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm 
# 5. class: 
# -- Iris Setosa 
# -- Iris Versicolour 
# -- Iris Virginica



def predict(model, input_df):
    predictions= model.predict(input_df)
    return predictions


st.title("Predict the Iris data type")
SL = st.number_input('sepal length in cm', min_value=0.0, max_value=5.0, value=1.5)
SW = st.number_input('sepal width in cm', min_value=0.0, max_value=5.0, value=1.5)
PL = st.number_input('petal length in cm ', min_value=0.0, max_value=5.0, value=1.5)
PW = st.number_input('petal width in cm ', min_value=0.0, max_value=5.0, value=1.5)

input_dict = {'SL' : SL, 'SW' : SW, 'PL' : PL, 'PW' : PW}
input_df = pd.DataFrame([input_dict])


output=[]

# Load the Model back from file
filename = 'knn.pkl'
with open(filename, 'rb') as file:  
      model = pickle.load(file)

        
        
st.subheader('Class Labels and their corresponding index number')
label_name = np.array(['Iris Setosa',
    'Iris Versicolour','Iris Virginica'])

st.write(label_name)                        
                        
if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = int(output[0])


st.success('The output is : {}'.format(label_name[output]))
    




# if __name__ == '__main__':
#     run()