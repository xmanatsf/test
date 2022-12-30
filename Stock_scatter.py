import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly           #(version 4.5.0)
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

url='https://github.com/xmanatsf/test/blob/main/RIY%20AND%20SP1500%20MONITORING%20SCREEN%20UPDATE4%2020221227.xlsx?raw=true'

data=pd.read_excel(url,engine='openpyxl')

sector=np.array(data['SEC'].values)
sec=np.unique(sector)

app = Dash(__name__)
server=app.server

app.layout = html.Div([
    html.H1(id = 'H1', children = 'factor scatter', style = {'textAlign':'center',\
                                            'marginTop':40,'marginBottom':40}),
     dcc.Dropdown( id = 'dropdown1',
        options = [{'label': x, 'value': x} for x in data.columns],
        value = None),
     
     dcc.Dropdown( id = 'dropdown2',
        options = [{'label': x, 'value': x} for x in data.columns],
        value = None),
     
     dcc.Dropdown( id = 'dropdown3',
     options=[{'label': x, 'value': x} for x in sec],
     value = None),
    # dcc.Graph(style={'height': '1000px'},id = 'scatter_plot', figure = fig_scatter (data,stk, var))
    dcc.Graph(style={'height': '700px'},id = 'scatter_plot')])

@app.callback(Output('scatter_plot', 'figure'),
              [Input("dropdown1", "value"), Input("dropdown2", "value"),Input("dropdown3", "value")])    

def update_chart(dropdown1_value,dropdown2_value,dropdown3_value):
        print(dropdown1_value,dropdown2_value,dropdown3_value)
        
        df=data[data['SEC'].isin([dropdown3_value])].dropna()
        size=np.log(df['mktcap'].values)*3
        fig = px.scatter(df, x=dropdown1_value, y=dropdown2_value
                         ,size=size, color="ticker")
        
        model = sm.OLS(df[dropdown2_value], sm.add_constant(df[dropdown1_value])).fit()
        print(model.summary())
        
        df['bestfit'] = sm.OLS(df[dropdown2_value], sm.add_constant(df[dropdown1_value])).fit().fittedvalues
        
        fig.add_trace(
            go.Scatter(x=df[dropdown1_value], y=df['bestfit'], name='trend',
                    line=dict(color='MediumPurple',width=2)))
        
        
        fig.update_layout(title = dropdown1_value+' vs '+ dropdown2_value,
                      xaxis_title = dropdown1_value,
                      yaxis_title = dropdown2_value
                      )
    
        return fig  



      
if __name__ == '__main__':
    app.run_server(debug=False)
    
