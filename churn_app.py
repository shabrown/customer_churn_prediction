import streamlit as st
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMClassifier

st.set_option('deprecation.showfileUploaderEncoding', False)

st.write("""
# Customer Churn Prediction App
### -----------------------by Sha Brown-------------------------
This app predicts whether a customer is going to churn!
Data obtained from the [Kaggle](https://www.kaggle.com/hassanamin/customer-churn)
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        age = st.sidebar.slider('Age', 22.0, 65.0, 42.0)
        total_purchase = st.sidebar.slider('Total Purchase', 100.0, 18026.0, 10046.0)
        account_manager = st.sidebar.selectbox('Account Manager', ('Yes', 'No'))
        years = st.sidebar.slider('Years', 1.0, 10.0, 5.5)
        num_sites = st.sidebar.slider('Number of Sites', 3.0, 14.0, 8.0)
        state = st.sidebar.selectbox('State', ('AL', 'AK', 'AZ', 'AR',
                                               'CA', 'CO', 'CT', 'DE',
                                               'FL', 'GA', 'HI', 'ID',
                                               'IL', 'IN', 'IA', 'KS',
                                               'KY', 'LA', 'ME', 'MD',
                                               'MA', 'MI', 'MN', 'MS',
                                               'MO', 'MT', 'NE', 'NV',
                                               'NH', 'NJ', 'NM', 'NY',
                                               'NC', 'ND', 'OH', 'OK',
                                               'OR', 'PA', 'RI', 'SC',
                                               'SD', 'TN', 'TX', 'UT',
                                               'VT', 'VA', 'WA', 'WV',
                                               'WI', 'WY', 'GU', 'PR',
                                               'VI')
                                     )

        data = {'age': age,
                'total_purchase': total_purchase,
                'account_manager': account_manager,
                'years': years,
                'num_sites': num_sites,
                'state': state
                }

        features = pd.DataFrame(data, index=[0])
        features['account_manager'] = features['account_manager'].apply(lambda x: 1 if x == 'Yes' else 0)
        features['purchase_per_year'] = features['total_purchase'] / features['years']
        features['sites_per_year'] = features['num_sites'] / features['years']
        features['purchase_per_site'] = features['total_purchase'] / features['num_sites']

        return features

input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
customers = pd.read_csv('customer_cleaned.csv')

state_churn_mean = customers.groupby('state')['churn'].mean().to_dict()

df = pd.concat([input_df, customers], axis=0)

df['state_churn'] = df['state'].map(state_churn_mean)

df = df.drop(columns=['state', 'churn'])

df = df[:1]  # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_rfc = pickle.load(open('churn_rfc.pkl', 'rb'))

# Apply model to make predictions
prediction = load_rfc.predict(df)
prediction_proba = load_rfc.predict_proba(df)

st.subheader('Will this customer stop using our service?')
predicted_outcome = np.array(['No', 'Yes'])
st.write(predicted_outcome[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
