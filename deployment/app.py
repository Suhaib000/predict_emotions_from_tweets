import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib
import nltk



    
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',dbc.themes.DARKLY]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def punctuation(val): 
  
    punctuations = '''()-[]{};:'"\,<>./@#$%^&_~'''
  
    for x in val.lower(): 
        if x in punctuations: 
            val = val.replace(x, " ") 
    return val

stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text


server = app.server

app.layout = html.Div(children=[
    dash.html.Br(),dash.html.Br(),
    html.Div([
        html.Div([html.H1('Emotions Prediction',style={'textAlign':'center','color':'white','font-size': '70px'})])
            ], className='row'),
    
dash.html.Br(),
dash.html.Br(),
    dash.html.Br(),dash.html.Br(),dash.html.Br(),
    html.Div([
     html.Div([] ,className='two columns'),
        
     html.Div([dcc.Textarea(
        id='textarea-state-example',
        style={'width': '100%', 'height': 75})], className='eight columns'),
        
     html.Div([], className='two columns'),
        ] ,className='row'),
    
dash.html.Br(),
    
    html.Div([ 
     html.Div([] ,className='two columns'),
     html.Div([html.Button('Submit', id='textarea-state-example-button', n_clicks=0),],className='eight columns'),
     html.Div([] ,className='two columns'),  
    ], className='row'),
    
dash.html.Br(),
    
     html.Div([
         html.Div([] ,className='five columns'), 
         html.Div([html.Div(id='textarea-state-example-output', style={'textAlign':'center',
                                                                       'whiteSpace': 'pre-line',
                                                                       'font-size': '20px'})] ,className='two columns'), 
         html.Div([] ,className='five columns'), 
     ],className='row')
    
    
    ])





@app.callback(
    Output('textarea-state-example-output', 'children'),
    Input('textarea-state-example-button', 'n_clicks'),
    State('textarea-state-example', 'value'))
def update_output(n_clicks, value):
    pipe_joblib = joblib.load('model.pkl')
    if n_clicks > 0:
        value=punctuation(value)
        value=stemm_text(value)
        result = pipe_joblib.predict([value]).item()
        if result=='happiness':
            result='Happy'
        return 'You Feel : \n{}'.format(result)

#======================================================================================================   
# app.run_server()
if __name__ == '__main__':
    app.run_server(debug=True)