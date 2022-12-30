# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:34:52 2022

@author: shibi
"""

import pandas as pd
# import matplotlib.pyplot as plt

import numpy as np
# import datetime
# from datetime import date

# import seaborn as sns

import plotly           #(version 4.5.0)
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

import io
import requests

from dash import Dash, dcc, html
from dash.dependencies import Input, Output

import os, signal
import warnings
warnings.filterwarnings("ignore")

# dt1=" 20221227"

# path= "C:\\Users\shibi\Documents\Research\screen\\"
# file_name="RIY AND SP1500 MONITORING SCREEN UPDATE4" + dt1 + ".xlsx"

# savepath= "C:\\Users\shibi\Documents\Research\screen\\"

# file= path + file_name

# data = pd.read_excel(file,engine='openpyxl')

url='https://github.com/xmanatsf/test/blob/main/RIY%20AND%20SP1500%20MONITORING%20SCREEN%20UPDATE4%2020221227.xlsx?raw=true'

# get_content = requests.get(url).content

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



# def update_scatter_chart(dropdown1_value,dropdown2_value,dropdown3_value):
    
#     print(dropdown1_value,dropdown2_value,dropdown3_value)
    
#     df=data[data['SEC'].isin([dropdown3_value])]
    
#     # df[['size1']] = StandardScaler().fit_transform(df[['mktcap']])
    
      
#     # colors=['red' if (i=='present') else 'blue' if (i=='new_only')
#     # else 'yellow' if (i=='old_only') else 'gray' for i in np.array(data['status'])]
    
#     # size=np.where((df['ticker'].isin(['AAPL','AMZN','MSFT','GOOGL','GOOG'])),df['mktcap']/(1e5),df['mktcap']/(1e3))
#     # size=np.array(df['size1'].values)
    
#     size=np.log(df['mktcap'].values)*3
    
#     fig = go.Figure([go.Scatter(mode='markers',x = df[dropdown1_value], 
#                       y = df[dropdown2_value],opacity=0.7,text = df['ticker'],
#                       hovertemplate=f"<b> ticker: %{{text}}</b><br><b>X: %{{x}}<br><b>Y:</b> %{{y}}<extra></extra>",
#                       marker=dict(color='green',size=size)
#                     )])
    
#     help_fig = px.scatter(df, x=dropdown1_value, y=dropdown2_value, trendline="ols")
#     # extract points as plain x and y
#     model = px.get_trendline_results(help_fig)
#     alpha = model.iloc[0]["px_fit_results"].params[0]
#     beta = model.iloc[0]["px_fit_results"].params[1]
    
#     # print(alpha.dtypes)
#     print(model.iloc[0]["px_fit_results"].rsquared)
#     print(model.iloc[0]["px_fit_results"].tvalues)
#     print(model.iloc[0]["px_fit_results"].params)
#     # print(alpha,beta)
    
#     df['bestfit'] = alpha + beta* df[dropdown1_value]
    
#     fig.add_trace(
#         go.Scatter(x=df[dropdown1_value], y=df['bestfit'], name='trend',
#                     line=dict(color='MediumPurple',width=2)))
    
#     print(df.head())
#     fig.update_traces(overwrite=True)
                                                     
#     # fig.data = [fig.data[0]]
#     # text = data['ticker'],hoverinfo = 'text',
#     # x=df[dropdown1_value]
#     # y=df[dropdown2_value]
#     # fig=px.scatter(df, x=x, y=y, size='mktcap',hover_data='ticker')
#     # Define updatemenus
     
#     fig.update_layout(title = dropdown1_value+' vs '+ dropdown2_value,
#                       xaxis_title = dropdown1_value,
#                       yaxis_title = dropdown2_value
#                       )
    
#     return fig  
      
if __name__ == '__main__':
    app.run_server(debug=False)
    
os.kill(os.getpid(), signal.SIGTERM)








# 'p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y','p_chg6m','p_chg3m','p_chg1m'
# ,'s_chg10y','s_chg7y','s_chg5y','s_chg3y','s_chg2y','s_chg1y','s_chg6m','s_chg3m','s_chg1m'
# ,'e_chg10y','e_chg7y','e_chg5y','e_chg3y','e_chg2y','e_chg1y','e_chg6m','e_chg3m','e_chg1m'
# ,'p_chg400d','p_chg300d','p_chg200d','p_chg150d','p_chg120d','p_chg70d','p_chg40d','p_chg20d'
# , 's_chg400d','s_chg300d','s_chg200d','s_chg150d','s_chg120d','s_chg70d','s_chg40d','s_chg20d'
# ,'e_chg400d','e_chg300d','e_chg200d','e_chg150d','e_chg120d','e_chg70d','e_chg40d','e_chg20d'
# ,'p_10y_cagr','p_7y_cagr','p_5y_cagr','p_3y_cagr','p_1y_cagr','s_10y_cagr',	's_7y_cagr','s_5y_cagr'	
# ,'s_3y_cagr','s_1y_cagr','e_10y_cagr',	'e_7y_cagr','e_5y_cagr','e_3y_cagr','e_1y_cagr'
# ,'p_tstat_10y','p_tstat_7y','p_tstat_5y','p_tstat_3y','p_tstat_2y','p_tstat_1y','p_tstat_6m','p_tstat_3m','p_tstat_1m'
# ,'s_tstat_10y','s_tstat_7y','s_tstat_5y','s_tstat_3y','s_tstat_2y','s_tstat_1y','s_tstat_6m','s_tstat_3m','s_tstat_1m'
# ,'e_tstat_10y','e_tstat_7y','e_tstat_5y','e_tstat_3y','e_tstat_2y','e_tstat_1y','e_tstat_6m','e_tstat_3m','e_tstat_1m'
# ,'p_tstat_200d','p_tstat_120d','p_tstat_50d','s_tstat_200d','s_tstat_120d','s_tstat_50d','e_tstat_200d','e_tstat_120d','e_tstat_50d'
