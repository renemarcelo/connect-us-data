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
    data = pd.read_csv('./data/joined_data_2.csv').drop(columns=['Unnamed: 0'])
    data.head()
#     cols = data.columns[1:]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(data[cols])
    
    # kmeans = KMeans(n_clusters=200)
    # kmeans.fit(X_scaled)

    # data['label'] = kmeans.labels_
    # y = data.label

#     knn_model = pickle.load(open('./model/knn_updated.pkl', 'rb'))
# #     knn_model.fit(X_scaled, y)
    
#     pref={'median_income': [500_000], 
#         'temp': [70], 
#         'total_population': [80_000],
#         'elev_in_ft': [80]}
    
#     prefdf = pd.DataFrame.from_dict(pref, orient='index').T
#     pref_scaled = scaler.transform(prefdf)
#     pred = knn_model.predict(pref_scaled)[0]
    
#     user_df = data[data.label == pred].county
#     user_show = pd.DataFrame(user_df)
#     return(user_show.to_dict())
#     return "<h1>Welcome to test</h1>"

if __name__ == "__main__":
    app.run(debug=True)
