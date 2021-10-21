from flask import Flask, Response, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# initialize the flask app
app = Flask(__name__)

# route 1: county dashboard
@app.route("/")

def county():
    np.random.seed(123)
    
    data = pd.read_csv('./data/joined_data_2.csv').drop(columns=['Unnamed: 0'])
    cols = data.columns[1:]
    print('line18')

    scaler = StandardScaler()
    print('line21')
    X_scaled = scaler.fit_transform(data[cols])
    print('line23')
    
    kmeans = KMeans(n_clusters=200)
    print('line26')
    kmeans.fit(X_scaled)
    print('line28')

    data['label'] = kmeans.labels_
    print('line31')
    y = data.label
    print('line33')

    knn_model = pickle.load(open('./model/knn_updated2.pkl', 'rb'))
    print('line36')
#     knn_model.fit(X_scaled, y)
    
    pref={'median_income': [500_000], 
        'temp': [70], 
        'total_population': [80_000],
        'elev_in_ft': [80]}
    print('line43')
    
    prefdf = pd.DataFrame.from_dict(pref, orient='index').T
    print('line46')
    pref_scaled = scaler.transform(prefdf)
    print('line48')
    pred = knn_model.predict(pref_scaled)[0]
    print('line50')
    
    user_df = data[data.label == pred].county
    print('line53')
    user_show = pd.DataFrame(user_df)
    print('line55')
    return(user_show.to_dict())
    print('line57')
    return "<h1>Welcome to test</h1>"

if __name__ == "__main__":
    app.run(debug=True)
