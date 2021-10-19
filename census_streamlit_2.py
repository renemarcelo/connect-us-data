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
user_pop_total = st.number_input('Population Total?')
user_unemployment_rate = st.number_input('Unemployment Rate?')
user_income = st.number_input('Median Income?')
user_temp = st.number_input('Climate (Temperature in Farenheit)?')

city = st.checkbox('find a place')

if city:
    data = pd.read_csv('./data/joined_data_2.csv').drop(columns=['Unnamed: 0'])
    cols = data.columns[1:]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[cols])

    kmeans = KMeans(n_clusters=1000)
    kmeans.fit(X_scaled)

    data['label'] = kmeans.labels_
    y = data.label

    knn = KNeighborsClassifier()
    knn.fit(X_scaled, y)

#     pref={'median_income': [80_000], 
#         'temp': [69], 
#         'total_population': [80_000],
#         'elev_in_ft': [80]}
    prefdf = pd.DataFrame.from_dict(pref)
    pref_scaled = scaler.transform(prefdf)

    pred = knn.predict(pref_scaled)[0]
    data[data.label == pred]
