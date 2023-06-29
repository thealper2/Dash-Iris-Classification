import dash
from dash import dcc, html
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

X = df.drop("target", axis=1)
y = df["target"]

with open("models/logreg.pkl", "rb") as f:
    model = pickle.load(f)

def predict_species(features):
    pred = model.predict([features])
    classes = ['setosa', 'versicolor', 'virginica']
    species = classes[int(pred[0])]
    return species

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Irisi Veri Seti EDA ve Model Tahmini"),
    dcc.Graph(id="feature-graph"),
    html.Div([
        html.H3("Özellik Değerleri"),
        html.Div([
            html.Label("Çanak yaprağı uzunluğu"),
            dcc.Slider(
                id="sepal-length-slider",
                min=X["sepal length (cm)"].min(),
                max=X["sepal length (cm)"].max(),
                step=0.1,
                value=X["sepal length (cm)"].mean()
            ),
            html.Label("Çanak yaprağı genişliği"),
            dcc.Slider(
                id="sepal-width-slider",
                min=X["sepal width (cm)"].min(),
                max=X["sepal width (cm)"].max(),
                step=0.1,
                value=X["sepal width (cm)"].mean()
            ),
            html.Label("Taç yaprağı uzunluğu"),
            dcc.Slider(
                id="petal-length-slider",
                min=X["petal length (cm)"].min(),
                max=X["petal length (cm)"].max(),
                step=0.1,
                value=X["petal length (cm)"].mean()
            ),
            html.Label("Taç yaprağı genişliği"),
            dcc.Slider(
                id="petal-width-slider",
                min=X["petal width (cm)"].min(),
                max=X["petal width (cm)"].max(),
                step=0.1,
                value=X["petal width (cm)"].mean()
            )
        ], style={"margin": "20px"})
    ]),
    html.Div(id="prediction-output", style={"margin": "20px"})
])

@app.callback(
    dash.dependencies.Output("feature-graph", "figure"),
    dash.dependencies.Input("sepal-length-slider", "value"),
    dash.dependencies.Input("sepal-width-slider", "value"),
    dash.dependencies.Input("petal-length-slider", "value"),
    dash.dependencies.Input("petal-width-slider", "value")
)

def update_feature_graph(sepal_length, sepal_width, petal_length, petal_width):
    fig = {
        'data': [
            {'x': X['sepal length (cm)'], 'y': X['sepal width (cm)'], 'mode': 'markers', 'name': 'Çanak yaprağı'},
            {'x': X['petal length (cm)'], 'y': X['petal width (cm)'], 'mode': 'markers', 'name': 'Taç yaprağı'}
        ],
        'layout': {
            'xaxis': {'title': 'Uzunluk (cm)'},
            'yaxis': {'title': 'Genişlik (cm)'},
            'hovermode': 'closest'
        }
    }


    return fig

@app.callback(
    dash.dependencies.Output("prediction-output", "children"),
    dash.dependencies.Input("sepal-length-slider", "value"),
    dash.dependencies.Input("sepal-width-slider", "value"),
    dash.dependencies.Input("petal-length-slider", "value"),
    dash.dependencies.Input("petal-width-slider", "value")
)

def update_prediction_output(sepal_length, sepal_width, petal_length, petal_width):
    species = predict_species([sepal_length, sepal_width, petal_length, petal_width])
    return f"Tahmin Edilen Tür: {species}"

if __name__ == "__main__":
    app.run_server(debug=True)