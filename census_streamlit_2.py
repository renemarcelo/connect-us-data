import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#Headers
st.header('Product Club')
st.subheader("US Census")

#Choose Action
user_median_income = st.number_input('Median Income?')
user_temp = st.number_input('Temperature? (F)')
user_total_population = st.number_input('Total Population?')
user_elev_in_ft = st.number_input('Elevation (ft)?')

city = st.checkbox('find a place')

if city:
    data = pd.read_csv('./data/joined_data_2.csv').drop(columns=['Unnamed: 0'])
    cols = data.columns[1:]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[cols])

#     kmeans = KMeans(n_clusters=1000)
    kmeans = pickle.load(open('./model/kmeans.pkl', 'rb'))
    kmeans.fit(X_scaled)

    data['label'] = kmeans.labels_
    y = data.label

#     knn = KNeighborsClassifier()
    knn_model = pickle.load(open('./model/knn_model.pkl', 'rb'))
    knn_model.fit(X_scaled, y)

    pref={'median_income': user_median_income, 
        'temp': user_temp, 
        'total_population': user_total_population,
        'elev_in_ft': user_elev_in_ft}
    
    prefdf = pd.DataFrame.from_dict(pref, orient='index').T
    pref_scaled = scaler.transform(prefdf)
    pred = knn_model.predict(pref_scaled)[0]
    
    user_df = data[data.label == pred].county
    user_show = pd.DataFrame(user_df)
    user_show
