import streamlit as st
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
import numpy

model = tf.keras.models.load_model('model.h5')
## load the encoder and scalar
with open('one_hot_encoder.pkl','rb') as file:
    one_hot_encoder = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
     label_encoder_gender= pickle.load(file)

with open('scalerl.pkl','rb') as file:
    scalerl = pickle.load(file)

## streamlit app
st.title('Customer Salary Prediction')

geography = st.selectbox(
    "Geography",
    one_hot_encoder.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider(
    "Age",
    18,
    92
)

balance = st.number_input(
    "Balance"
)

credit_score = st.number_input(
    "Credit Score"
)

exited = st.number_input(
    "Exited"
)

tenure = st.slider(
    "Tenure",
    0,
    10
)

num_of_products = st.slider(
    "Number of Products",
    1,
    4
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One_hot_encoder "Geography"
hot_encoded= one_hot_encoder.transform([[geography]]).toarray()
hot_encoded_df= pd.DataFrame(hot_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

# combine one_hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), hot_encoded_df],axis=1)

# scale the input data
input_data_scaled = scalerl.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_value=prediction[0][0]

st.write(f'model.predict: {prediction_value:.2f}')
