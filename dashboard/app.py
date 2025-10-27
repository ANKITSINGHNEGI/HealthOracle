import dash
from dash import html, dcc
app = dash.Dash(__name__)
app.layout = html.Div([html.H3("HealthOracle Dashboard"), dcc.Graph(id="risk-gauges"), dcc.Graph(id="shap-summary"), dcc.Graph(id="population-compare")])
server = app.server
