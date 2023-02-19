# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:34:52 2022

@author: shibi
"""

import pandas as pd
import numpy as np
import datetime
from datetime import date
import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import statsmodels.api as sm

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

# SEC='RBICS_SEC2'

# sector=np.array(data[SEC].values)
# sec=np.unique(sector)

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__)
# server=app.server

app.layout = html.Div([
    html.H1(id = 'H1', children = 'factor scatter', style = {'textAlign':'center',\
                                            'marginTop':40,'marginBottom':40}),## Title line
   
       
     html.Div(children=[
         dcc.Dropdown( id = 'dropdown1',
        options = ['p_chg200d','p_chg120d','p_chg70d','p_chg40d','p_chg20d',
                   'p_tstat_200d','p_tstat_120d','p_tstat_70d','p_tstat_40d','p_tstat_20d'],
        value = None, style={'width': '25%', 'display': 'inline-block'}),## drop down select x variable
     
     dcc.Dropdown( id = 'dropdown2',
        options = ['p_chg200d','p_chg120d','p_chg70d','p_chg40d','p_chg20d',
                   'p_tstat_200d','p_tstat_120d','p_tstat_70d','p_tstat_40d','p_tstat_20d'],
        value = None, style={'width': '25%', 'display': 'inline-block'}),
     
     dcc.Dropdown( id = 'dropdown3',
        options = ['SEC', 'INDG', 'IND', 'RBICS_SEC','RBICS_SEC1','RBICS_SEC2'],
        value = None, style={'width': '25%', 'display': 'inline-block'}),
     
     dcc.Dropdown( id = 'dropdown4',
        options = ['20230216','20230206','20230121'],
        value = None, style={'width': '25%', 'display': 'inline-block'})
     
     ]),## drop down select y variable

    html.Div(children=[dcc.Slider(min=6, max=23, step=1, value=12, id='myslider')
    ]),
    
    html.Div([
        dcc.Graph(id = 'scatter_plot0',clickData={'points': [{'text': 'nvda'}]}
         ,style={'width': '98%','height': '700px','display': 'inline-block'})
        ]),  

      html.Div(children=[
         dcc.Graph(id = 'scatter_plot1',clickData={'points': [{'text': 'nvda'}]}
          ,style={'width': '98%','height': '700px','display': 'inline-block'}),  #hoverData={'points': [{'ticker': 'nvda'}]}  sector specific scatter plot
         ]),
     
      html.Div(children=[
        dcc.Graph(id = 'ts_plot1',clickData={'points': [{'text': 'nvda'}]}
         ,style={'width': '49%','height': '500px','display': 'inline-block'}),
        
        dcc.Graph(id = 'ts_plot2',clickData={'points': [{'text': 'nvda'}]}
          ,style={'width': '49%','height': '500px','display': 'inline-block'})
    ]),
     
      html.Div([
          dcc.Graph(id = 'scatter_plot2',clickData={'points': [{'text': 'nvda'}]}
          ,style={'width': '98%','height': '700px','display': 'inline-block'}),  #hoverData={'points': [{'ticker': 'nvda'}]}  sector specific scatter plot
          ]),

    html.Div(children=[
        dcc.Graph(id = 'ts_plot3',clickData={'points': [{'text': 'nvda'}]}
          ,style={'width': '49%','height': '500px','display': 'inline-block'}),
        
        dcc.Graph(id = 'ts_plot4',clickData={'points': [{'text': 'nvda'}]}
          ,style={'width': '49%','height': '500px','display': 'inline-block'})
    ]),    
    
    html.Div(children=[
        dcc.Graph(id = 'ts_plot5',clickData={'points': [{'text': 'nvda'}]}
          ,style={'width': '49%','height': '500px','display': 'inline-block'}),
        dcc.Graph(id = 'ts_plot6',clickData={'points': [{'text': 'nvda'}]}
          ,style={'width': '49%','height': '500px','display': 'inline-block'})
    ]),    

   
    # dcc.Store stores the intermediate value
    
    html.Div(children=[dcc.Store(id='Scatter_data'),
                       dcc.Store(id='TS_data')])
  
])

# =============================================================================
# store fetched data in the odyssey_data     
# =============================================================================


@app.callback(Output('Scatter_data', 'data'),
              Input("dropdown4", "value"))
def odyssey_data(dropdown4_value):

    dt1=dropdown4_value
    url_start = "https://github.com/xmanatsf/test/blob/main/US%20SCREEN%20WITH%20UNIVERSAL%20VARIABLE%20"
    url_end = ".xlsx?raw=true"
    query= url_start + dt1 + url_end
    df=pd.read_excel(query,engine='openpyxl')
   
    # print(df.head())
    
     # more generally, this line would be
     # json.dumps(cleaned_df)
    return df.to_json(date_format='iso', orient='split') 


########################

@app.callback(Output('TS_data', 'data'),
              Input("dropdown4", "value"))
def odyssey_data(dropdown4_value):
    # print(dropdown4_value)
    
    dt1=dropdown4_value
    url_start = "https://github.com/xmanatsf/test/blob/main/US%20Screen%20monthly%20with%20universal%20variable2%20"
    url_end = ".xlsx?raw=true"
    query= url_start + dt1 + url_end
    df=pd.read_excel(query,engine='openpyxl')
   

    return df.to_json(date_format='iso', orient='split') 

    
    return df.to_json(date_format='iso', orient='split')    


# =============================================================================
# add sector level scatter plot    
# =============================================================================
    
@app.callback(Output('scatter_plot0', 'figure'),
              [Input("dropdown3", "value"),Input('Scatter_data', 'data')])    

def update_chart1(dropdown3_value,jsonified_cleaned_data):
        # print(dropdown1_value,dropdown2_value,dropdown3_value)
        
        data = pd.read_json(jsonified_cleaned_data, orient='split')
        
       
       
        period=['70d','40d','20d']
        
        fig = make_subplots(rows=1, cols=3)
          
        for i in range(len(period)): 
            sig='p_tstat_'+period[i]
            
            dtemp=data[['p_tstat_200d', sig,dropdown3_value,'mktcap']]
        
            df1=dtemp.groupby([dropdown3_value]).agg(
                x_var=('p_tstat_200d', 'mean'),
                y_var=(sig, 'mean'), 
                mktcap=('mktcap', 'mean')
                )
            
            df=df1[df1.index!='@NA']
            df=df.reset_index()
            df=df.dropna(subset=['mktcap'])
            df['ind']=df[dropdown3_value]
            
            dtemp=df[['x_var', 'y_var','ind']]
            
            # print(dtemp.head())
        
            size=np.log(df['mktcap'].values)*3
            
            if i<1:     
                fig.add_trace(
                    go.Scatter(x=df['x_var'], y=df['y_var'], text=df['ind'],opacity=0.5,
                        mode="markers+text",marker=dict(color='LightSkyBlue',size=10)
                        ),row=1, col=1)
            
                # px.scatter(df, x='x_var', y='y_var'
                #              ,size=size, color='ind', text='ind')
        
                model = sm.OLS(df['y_var'], sm.add_constant(df['x_var'])).fit()
                # print(model.summary())
                
                df['bestfit'] = sm.OLS(df['y_var'], sm.add_constant(df['x_var'])).fit().fittedvalues
                
                fig.add_trace(
                    go.Scatter(x=df['x_var'], y=df['bestfit'], name='trend',
                            line=dict(color='MediumPurple',width=2)),row=1, col=1)
             

                
            elif i==1:     
                fig.add_trace(
                    go.Scatter(x=df['x_var'], y=df['y_var'], text=df['ind'],opacity=0.5,
                        mode="markers+text",marker=dict(color='LightSkyBlue',size=10)
                        ),row=1, col=2)
            
                # px.scatter(df, x='x_var', y='y_var'
                #              ,size=size, color='ind', text='ind')
        
                model = sm.OLS(df['y_var'], sm.add_constant(df['x_var'])).fit()
                # print(model.summary())
                
                df['bestfit'] = sm.OLS(df['y_var'], sm.add_constant(df['x_var'])).fit().fittedvalues
                
                fig.add_trace(
                    go.Scatter(x=df['x_var'], y=df['bestfit'], name='trend',
                            line=dict(color='MediumPurple',width=2)),row=1, col=2)
                


            else:     
                fig.add_trace(
                    go.Scatter(x=df['x_var'], y=df['y_var'], text=df['ind'],opacity=0.5,
                        mode="markers+text",marker=dict(color='LightSkyBlue',size=10)
                        ),row=1, col=3)
            
                # px.scatter(df, x='x_var', y='y_var'
                #              ,size=size, color='ind', text='ind')
        
                model = sm.OLS(df['y_var'], sm.add_constant(df['x_var'])).fit()
                # print(model.summary())
                
                df['bestfit'] = sm.OLS(df['y_var'], sm.add_constant(df['x_var'])).fit().fittedvalues
                
                fig.add_trace(
                    go.Scatter(x=df['x_var'], y=df['bestfit'], name='trend',
                            line=dict(color='MediumPurple',width=2)),row=1, col=3)
                
 
                
            fig.update_layout(updatemenus=[
                                    dict(
                                        type = "buttons",
                                        direction = "left",
                                        buttons=list([
                                            dict(
                                                args=["visible", "legendonly"],
                                                label="Deselect All",
                                                method="restyle"
                                            ),
                                            dict(
                                                args=["visible", True],
                                                label="Select All",
                                                method="restyle"
                                            )
                                        ]),
                                        pad={"r": 10, "t": 10},
                                        showactive=False,
                                        x=1,
                                        xanchor="right",
                                        y=1.1,
                                        yanchor="top"
                                    ),
                                    ],
                title = 'p_tstat_200d vs selected p_tsat',
                          xaxis_title = 'p_tstat_200d')
    
        return fig      

# =============================================================================
# add sector level scatter plot    
# =============================================================================
    
@app.callback(Output('scatter_plot1', 'figure'),
              [Input("dropdown1", "value"), Input("dropdown2", "value")
               , Input("dropdown3", "value"),Input('Scatter_data', 'data')])    

def update_chart1(dropdown1_value,dropdown2_value,dropdown3_value,jsonified_cleaned_data):
        # print(dropdown1_value,dropdown2_value,dropdown3_value)
        
        data = pd.read_json(jsonified_cleaned_data, orient='split')
        
        df=data[[dropdown3_value, 'ticker',dropdown1_value,dropdown2_value,'mktcap']]
        
        # print(df.head())
        
        df1=df.groupby([dropdown3_value]).agg(
            x_var=(dropdown1_value, 'mean'),
            y_var=(dropdown2_value, 'mean'), 
            mktcap=('mktcap', 'mean')
            )
        
        df=df1[df1.index!='@NA']
        df=df.reset_index()
        
        # print(df.head())
        
              
        df=df.dropna(subset=['mktcap'])
        df['ind']=df[dropdown3_value]
        
        dtemp=data[[dropdown1_value,dropdown2_value]].median()
        dtemp=dtemp.reset_index()
        
        # print(dtemp.head())
        # print(dtemp.loc[0, :].values.tolist())
        
        size=np.log(df['mktcap'].values)*3
        fig = px.scatter(df, x='x_var', y='y_var'
                         ,size=size, color='ind', text='ind')
        
        model = sm.OLS(df['y_var'], sm.add_constant(df['x_var'])).fit()
        # print(model.summary())
        
        df['bestfit'] = sm.OLS(df['y_var'], sm.add_constant(df['x_var'])).fit().fittedvalues
        
        fig.add_trace(
            go.Scatter(x=df['x_var'], y=df['bestfit'], name='trend',
                    line=dict(color='MediumPurple',width=2)))
        
        # fig.add_trace(
        #     go.Scatter(mode='markers',x=df['x_var'], y=df['y_var'], name='all',
        #                 opacity=0.3,text = df['ind'],
        #             hovertemplate=f"<b> ind: %{{text}}</b><br><b>X: %{{x}}<br><b>Y:</b> %{{y}}<extra></extra>",
        #             marker=dict(color='gray',size=10)))
        
        fig.add_trace(
            go.Scatter(mode='markers',x=dtemp.loc[0, :].values.tolist()
                       , y=dtemp.loc[1, :].values.tolist(), name='Mkt',
                       text = 'mkt',
                    hovertemplate=f"<b> mkt </b><br><b>X: %{{x}}<br><b>Y:</b> %{{y}}<extra></extra>",
                    marker=dict(color='darkred',size=30)))
        
        fig.update_layout(updatemenus=[
                                dict(
                                    type = "buttons",
                                    direction = "left",
                                    buttons=list([
                                        dict(
                                            args=["visible", "legendonly"],
                                            label="Deselect All",
                                            method="restyle"
                                        ),
                                        dict(
                                            args=["visible", True],
                                            label="Select All",
                                            method="restyle"
                                        )
                                    ]),
                                    pad={"r": 10, "t": 10},
                                    showactive=False,
                                    x=1,
                                    xanchor="right",
                                    y=1.1,
                                    yanchor="top"
                                ),
                                ],
            title = dropdown1_value+' vs '+ dropdown2_value,
                      xaxis_title = dropdown1_value,
                      yaxis_title = dropdown2_value
                      )
    
        return fig      

# =============================================================================
# Ind charts
# =============================================================================

@app.callback(Output('ts_plot1', 'figure'),
              [Input("scatter_plot1", "clickData"), Input("dropdown3", "value")
              ,Input('TS_data', 'data'),Input('myslider', 'value')])    

def update_timeseries1(clickData,dropdown3_value,jsonified_cleaned_data,myslider):
    
    # print(clickData['points'][0])
    
    ind=clickData['points'][0]['text']
    
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    
    df=data[data[dropdown3_value].isin([clickData['points'][0]['text']])]
    
    # print(df.head())
    
   
    dtemp=df[[dropdown3_value, 's_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11',  's_chg_12','s_chg_13','s_chg_14','s_chg_15','s_chg_16','s_chg_17','s_chg_18','s_chg_19'
          ,'s_chg_20','s_chg_21','s_chg_22','s_chg_23'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11',  'e_chg_12','e_chg_13','e_chg_14','e_chg_15','e_chg_16','e_chg_17','e_chg_18','e_chg_19'
          ,'e_chg_20','e_chg_21','e_chg_22','e_chg_23'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11',  'p_chg_12','p_chg_13','p_chg_14','p_chg_15','p_chg_16','p_chg_17','p_chg_18','p_chg_19'
          ,'p_chg_20','p_chg_21','p_chg_22','p_chg_23']]
    
 
    dtemp=dtemp.groupby([dropdown3_value]).agg(
        s0=('s_chg_0', 'mean'),s1=('s_chg_1', 'mean'),s2=('s_chg_2', 'mean'),s3=('s_chg_3', 'mean'),
        s4=('s_chg_4', 'mean'),s5=('s_chg_5', 'mean'),s6=('s_chg_6', 'mean'),s7=('s_chg_7', 'mean'),
        s8=('s_chg_8', 'mean'),s9=('s_chg_9', 'mean'),s10=('s_chg_10', 'mean'),s11=('s_chg_11', 'mean'),
        s12=('s_chg_12', 'mean'),s13=('s_chg_13', 'mean'),s14=('s_chg_14', 'mean'),s15=('s_chg_15', 'mean'),
        s16=('s_chg_16', 'mean'),s17=('s_chg_17', 'mean'),s18=('s_chg_18', 'mean'),s19=('s_chg_19', 'mean'),
        s20=('s_chg_20', 'mean'),s21=('s_chg_21', 'mean'),s22=('s_chg_22', 'mean'),s23=('s_chg_23', 'mean'),
        e0=('e_chg_0', 'mean'),e1=('e_chg_1', 'mean'),e2=('e_chg_2', 'mean'),e3=('e_chg_3', 'mean'),
        e4=('e_chg_4', 'mean'),e5=('e_chg_5', 'mean'),e6=('e_chg_6', 'mean'),e7=('e_chg_7', 'mean'),
        e8=('e_chg_8', 'mean'),e9=('e_chg_9', 'mean'),e10=('e_chg_10', 'mean'),e11=('e_chg_11', 'mean'),
        e12=('e_chg_12', 'mean'),e13=('e_chg_13', 'mean'),e14=('e_chg_14', 'mean'),e15=('e_chg_15', 'mean'),
        e16=('e_chg_16', 'mean'),e17=('e_chg_17', 'mean'),e18=('e_chg_18', 'mean'),e19=('e_chg_19', 'mean'),
        e20=('e_chg_20', 'mean'),e21=('e_chg_21', 'mean'),e22=('e_chg_22', 'mean'),e23=('e_chg_23', 'mean'),
        p0=('p_chg_0', 'mean'),p1=('p_chg_1', 'mean'),p2=('p_chg_2', 'mean'),p3=('p_chg_3', 'mean'),
        p4=('p_chg_4', 'mean'),p5=('p_chg_5', 'mean'),p6=('p_chg_6', 'mean'),p7=('p_chg_7', 'mean'),
        p8=('p_chg_8', 'mean'),p9=('p_chg_9', 'mean'),p10=('p_chg_10', 'mean'),p11=('p_chg_11', 'mean'),
        p12=('p_chg_12', 'mean'),p13=('p_chg_13', 'mean'),p14=('p_chg_14', 'mean'),p15=('p_chg_15', 'mean'),
        p16=('p_chg_16', 'mean'),p17=('p_chg_17', 'mean'),p18=('p_chg_18', 'mean'),p19=('p_chg_19', 'mean'),
        p20=('p_chg_20', 'mean'),p21=('p_chg_21', 'mean'),p22=('p_chg_22', 'mean'),p23=('p_chg_23', 'mean'))
    
    dtemp=dtemp[dtemp.index!='@NA']
    dtemp=dtemp.reset_index()
      
    df_s1=dtemp[['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11',
                 's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23']]
    df_s1=df_s1.rename({'s0': '0', 's1': '1','s2': '2',
                        's3': '3', 's4': '4','s5': '5',
                        's6': '6', 's7': '7', 's8': '8', 's9': '9'
                        , 's10': '10', 's11': '11',
                        's12': '12', 's13': '13','s14': '14',
                        's15': '15', 's16': '16','s17': '17',
                        's18': '18', 's19': '19', 's20': '20', 's21': '21'
                        , 's22': '22', 's23': '23'}
                        , axis=1)
    
       
    df_s2=dtemp[['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11',
                 'e12','e13','e14','e15','e16','e17','e18','e19','e20','e21','e22','e23']]
    df_s2=df_s2.rename({'e0': '0', 'e1': '1','e2': '2',
                        'e3': '3', 'e4': '4','e5': '5',
                        'e6': '6', 'e7': '7', 'e8': '8', 'e9': '9'
                        , 'e10': '10', 'e11': '11',
                        'e12': '12', 'e13': '13','e14': '14',
                        'e15': '15', 'e16': '16','e17': '17',
                        'e18': '18', 'e19': '19', 'e20': '20', 'e21': '21'
                        , 'e22': '22', 'e23': '23'}
                        , axis=1)
    
    df_s3=dtemp[['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11',
                 'p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23']]
    df_s3=df_s3.rename({'p0': '0', 'p1': '1','p2': '2',
                        'p3': '3', 'p4': '4','p5': '5',
                        'p6': '6', 'p7': '7', 'p8': '8', 'p9': '9'
                        , 'p10': '10', 'p11': '11',
                        'p12': '12', 'p13': '13','p14': '14',
                        'p15': '15', 'p16': '16','p17': '17',
                        'p18': '18', 'p19': '19', 'p20': '20', 'p21': '21'
                        , 'p22': '22', 'p23': '23'}
                        , axis=1)
    var=['s','e','p']
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    df1['period1'] = df1['period'].astype(int)
    df1 = df1[df1['period1'] <= myslider]
    df1=df1.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    df1['cum']= df1[['Rate']].cumsum()
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    df2['period1'] = df2['period'].astype(int)
    df2 = df2[df2['period1'] <= myslider]
    df2=df2.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    df2['cum']= df2[['Rate']].cumsum()
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3['period1'] = df3['period'].astype(int)
    df3 = df3[df3['period1'] <= myslider]
    df3=df3.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    df3['cum']= df3[['Rate']].cumsum()
    
    df = df1.append(df2)
    df = df.append(df3)
    # print(df.head())
       
    dmkt=data[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11',  's_chg_12','s_chg_13','s_chg_14','s_chg_15','s_chg_16','s_chg_17','s_chg_18','s_chg_19'
          ,'s_chg_20','s_chg_21','s_chg_22','s_chg_23'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11',  'e_chg_12','e_chg_13','e_chg_14','e_chg_15','e_chg_16','e_chg_17','e_chg_18','e_chg_19'
          ,'e_chg_20','e_chg_21','e_chg_22','e_chg_23'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11',  'p_chg_12','p_chg_13','p_chg_14','p_chg_15','p_chg_16','p_chg_17','p_chg_18','p_chg_19'
          ,'p_chg_20','p_chg_21','p_chg_22','p_chg_23']].mean()
    
    dmkt=dmkt.reset_index()

    dmkt.rename(columns={'index':'period',0:'Rate'}, inplace = True)
    dmkt_s=dmkt[dmkt['period'].isin(['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11',  's_chg_12','s_chg_13','s_chg_14','s_chg_15','s_chg_16','s_chg_17','s_chg_18','s_chg_19'
          ,'s_chg_20','s_chg_21','s_chg_22','s_chg_23'])]
    dmkt_s['period']=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'
                      , '16', '17', '18', '19', '20', '21', '22', '23']
    dmkt_s['period1'] = dmkt_s['period'].astype(int)
    dmkt_s = dmkt_s[dmkt_s['period1'] <= myslider]
    dmkt_s=dmkt_s.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    dmkt_s['cum']= dmkt_s[['Rate']].cumsum()    
    
    dmkt_e=dmkt[dmkt['period'].isin(['e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
   ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11',  'e_chg_12','e_chg_13','e_chg_14','e_chg_15','e_chg_16','e_chg_17','e_chg_18','e_chg_19'
      ,'e_chg_20','e_chg_21','e_chg_22','e_chg_23'])]
    dmkt_e['period']=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'
                      , '16', '17', '18', '19', '20', '21', '22', '23']
    dmkt_e['period1'] = dmkt_e['period'].astype(int)
    dmkt_e = dmkt_e[dmkt_e['period1'] <= myslider]
    dmkt_e=dmkt_e.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    dmkt_e['cum']= dmkt_e[['Rate']].cumsum()   
    
    dmkt_p=dmkt[dmkt['period'].isin(['p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
    ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11',  'p_chg_12','p_chg_13','p_chg_14','p_chg_15','p_chg_16','p_chg_17','p_chg_18','p_chg_19'
       ,'p_chg_20','p_chg_21','p_chg_22','p_chg_23'])]
    dmkt_p['period']=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'
                      , '16', '17', '18', '19', '20', '21', '22', '23']
    dmkt_p['period1'] = dmkt_p['period'].astype(int)
    dmkt_p = dmkt_p[dmkt_p['period1'] <= myslider]
    dmkt_p=dmkt_p.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    dmkt_p['cum']= dmkt_p[['Rate']].cumsum()       

    dmkt_s['indicator']=var[0]
    dmkt_e['indicator']=var[1]
    dmkt_p['indicator']=var[2]
        
        
    dmkt = dmkt_s.append(dmkt_e)
    dmkt = dmkt.append(dmkt_p)
    
    # print(dmkt.head())    

    fig = make_subplots(rows=1, cols=1,specs=[[{"secondary_y": True}]])


    colors = ['red','blue','purple','darkcyan','darkred']

    for i in range(len(var)): 
        sig=var[i]
        
        df_temp=df[df['indicator'].isin([sig])]
        
        dmkt_temp=dmkt[dmkt['indicator'].isin([sig])]
        
        if i<2:     
            fig.add_trace(
                go.Scatter(x=df_temp.period,
                           y=df_temp.cum,
                           name=sig,
                            visible=True,
                            line=dict(width=1,color=colors[i]),
                        showlegend=True),row=1, col=1)
            
            fig.add_trace(
                go.Scatter(x=dmkt_temp.period,
                           y=dmkt_temp.cum,
                           name='mkt '+sig,
                            visible=True,
                            line=dict(width=1,color=colors[i],dash='dot'),
                        showlegend=True),row=1, col=1)
            
        else:
            fig.add_trace(
                go.Scatter(x=df_temp.period,
                           y=df_temp.cum,
                           name=sig,
                            visible=True,
                            line=dict(width=1,color=colors[i]),
                        showlegend=True),secondary_y=True,row=1, col=1)
            fig.add_trace(
                go.Scatter(x=dmkt_temp.period,
                           y=dmkt_temp.cum,
                           name='mkt '+sig,
                            visible=True,
                            line=dict(width=1,color=colors[i],dash='dot'),
                        showlegend=True),secondary_y=True,row=1, col=1)

    fig.update_layout(title = ind+' cum s e and p trend',
                  xaxis_title = 'DATE',
                  yaxis_title = 'trend',
    paper_bgcolor="LightSteelBlue",
    legend=dict(orientation="h",
    yanchor="bottom",
    y=-0.2,
    xanchor="center",
    x=0.5))             
    
    return fig
       
###################

@app.callback(Output('ts_plot2', 'figure'),
              [Input("scatter_plot1", "clickData"), Input("dropdown3", "value")
              ,Input('TS_data', 'data'),Input('myslider', 'value')])    

def update_timeseries2(clickData,dropdown3_value,jsonified_cleaned_data,myslider):
    
    
    ind=clickData['points'][0]['text']
    
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    
    df=data[data[dropdown3_value].isin([clickData['points'][0]['text']])]
    
    # print(df.head())
    
   
    dtemp=df[[dropdown3_value, 's_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11',  's_chg_12','s_chg_13','s_chg_14','s_chg_15','s_chg_16','s_chg_17','s_chg_18','s_chg_19'
          ,'s_chg_20','s_chg_21','s_chg_22','s_chg_23'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11',  'e_chg_12','e_chg_13','e_chg_14','e_chg_15','e_chg_16','e_chg_17','e_chg_18','e_chg_19'
          ,'e_chg_20','e_chg_21','e_chg_22','e_chg_23'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11',  'p_chg_12','p_chg_13','p_chg_14','p_chg_15','p_chg_16','p_chg_17','p_chg_18','p_chg_19'
          ,'p_chg_20','p_chg_21','p_chg_22','p_chg_23']]
    
 
    dtemp=dtemp.groupby([dropdown3_value]).agg(
        s0=('s_chg_0', 'mean'),s1=('s_chg_1', 'mean'),s2=('s_chg_2', 'mean'),s3=('s_chg_3', 'mean'),
        s4=('s_chg_4', 'mean'),s5=('s_chg_5', 'mean'),s6=('s_chg_6', 'mean'),s7=('s_chg_7', 'mean'),
        s8=('s_chg_8', 'mean'),s9=('s_chg_9', 'mean'),s10=('s_chg_10', 'mean'),s11=('s_chg_11', 'mean'),
        s12=('s_chg_12', 'mean'),s13=('s_chg_13', 'mean'),s14=('s_chg_14', 'mean'),s15=('s_chg_15', 'mean'),
        s16=('s_chg_16', 'mean'),s17=('s_chg_17', 'mean'),s18=('s_chg_18', 'mean'),s19=('s_chg_19', 'mean'),
        s20=('s_chg_20', 'mean'),s21=('s_chg_21', 'mean'),s22=('s_chg_22', 'mean'),s23=('s_chg_23', 'mean'),
        e0=('e_chg_0', 'mean'),e1=('e_chg_1', 'mean'),e2=('e_chg_2', 'mean'),e3=('e_chg_3', 'mean'),
        e4=('e_chg_4', 'mean'),e5=('e_chg_5', 'mean'),e6=('e_chg_6', 'mean'),e7=('e_chg_7', 'mean'),
        e8=('e_chg_8', 'mean'),e9=('e_chg_9', 'mean'),e10=('e_chg_10', 'mean'),e11=('e_chg_11', 'mean'),
        e12=('e_chg_12', 'mean'),e13=('e_chg_13', 'mean'),e14=('e_chg_14', 'mean'),e15=('e_chg_15', 'mean'),
        e16=('e_chg_16', 'mean'),e17=('e_chg_17', 'mean'),e18=('e_chg_18', 'mean'),e19=('e_chg_19', 'mean'),
        e20=('e_chg_20', 'mean'),e21=('e_chg_21', 'mean'),e22=('e_chg_22', 'mean'),e23=('e_chg_23', 'mean'),
        p0=('p_chg_0', 'mean'),p1=('p_chg_1', 'mean'),p2=('p_chg_2', 'mean'),p3=('p_chg_3', 'mean'),
        p4=('p_chg_4', 'mean'),p5=('p_chg_5', 'mean'),p6=('p_chg_6', 'mean'),p7=('p_chg_7', 'mean'),
        p8=('p_chg_8', 'mean'),p9=('p_chg_9', 'mean'),p10=('p_chg_10', 'mean'),p11=('p_chg_11', 'mean'),
        p12=('p_chg_12', 'mean'),p13=('p_chg_13', 'mean'),p14=('p_chg_14', 'mean'),p15=('p_chg_15', 'mean'),
        p16=('p_chg_16', 'mean'),p17=('p_chg_17', 'mean'),p18=('p_chg_18', 'mean'),p19=('p_chg_19', 'mean'),
        p20=('p_chg_20', 'mean'),p21=('p_chg_21', 'mean'),p22=('p_chg_22', 'mean'),p23=('p_chg_23', 'mean'))
    
    dtemp=dtemp[dtemp.index!='@NA']
    dtemp=dtemp.reset_index()
      
    df_s1=dtemp[['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11',
                 's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23']]
    df_s1=df_s1.rename({'s0': '0', 's1': '1','s2': '2',
                        's3': '3', 's4': '4','s5': '5',
                        's6': '6', 's7': '7', 's8': '8', 's9': '9'
                        , 's10': '10', 's11': '11',
                        's12': '12', 's13': '13','s14': '14',
                        's15': '15', 's16': '16','s17': '17',
                        's18': '18', 's19': '19', 's20': '20', 's21': '21'
                        , 's22': '22', 's23': '23'}
                        , axis=1)
    
    df_s2=dtemp[['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11',
                 'e12','e13','e14','e15','e16','e17','e18','e19','e20','e21','e22','e23']]
    df_s2=df_s2.rename({'e0': '0', 'e1': '1','e2': '2',
                        'e3': '3', 'e4': '4','e5': '5',
                        'e6': '6', 'e7': '7', 'e8': '8', 'e9': '9'
                        , 'e10': '10', 'e11': '11',
                        'e12': '12', 'e13': '13','e14': '14',
                        'e15': '15', 'e16': '16','e17': '17',
                        'e18': '18', 'e19': '19', 'e20': '20', 'e21': '21'
                        , 'e22': '22', 'e23': '23'}
                        , axis=1)
    
    df_s3=dtemp[['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11',
                 'p12','p13','p14','p15','p16','p17','p18','p19','p20','p21','p22','p23']]
    df_s3=df_s3.rename({'p0': '0', 'p1': '1','p2': '2',
                        'p3': '3', 'p4': '4','p5': '5',
                        'p6': '6', 'p7': '7', 'p8': '8', 'p9': '9'
                        , 'p10': '10', 'p11': '11',
                        'p12': '12', 'p13': '13','p14': '14',
                        'p15': '15', 'p16': '16','p17': '17',
                        'p18': '18', 'p19': '19', 'p20': '20', 'p21': '21'
                        , 'p22': '22', 'p23': '23'}
                        , axis=1)
    var=['s','e','p']
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    df1['period1'] = df1['period'].astype(int)
    df1 = df1[df1['period1'] <= myslider]
    df1=df1.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))

    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    df2['period1'] = df2['period'].astype(int)
    df2 = df2[df2['period1'] <= myslider]
    df2=df2.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))

    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3['period1'] = df3['period'].astype(int)
    df3 = df3[df3['period1'] <= myslider]
    df3=df3.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))


    df = df1.append(df2)
    df = df.append(df3)
    
    # print(df.head())
       
   
    fig = make_subplots(rows=1, cols=1,specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df1.period, y=df1.Rate, fill='tozeroy',
                       name='s', mode='none' # override default markers+lines
                        ))
    fig.add_trace(go.Scatter(x=df3.period, y=df3.Rate, fill='tonexty',
                       name='p', mode= 'none'),secondary_y=True,)
    
    fig.update_layout(
    paper_bgcolor="LightSteelBlue",
    legend=dict(orientation="h",
    yanchor="bottom",
    y=-0.2,
    xanchor="center",
    x=0.5))
    
    return fig



# =============================================================================
# 
# =============================================================================
@app.callback(Output('scatter_plot2', 'figure'),
              [Input("dropdown1", "value"), Input("dropdown2", "value")
               , Input("dropdown3", "value"),Input("scatter_plot1", "clickData")
               ,Input('Scatter_data', 'data')])    

def update_chart2(dropdown1_value,dropdown2_value,dropdown3_value,clickData,jsonified_cleaned_data):
        # print(dropdown1_value,dropdown2_value,dropdown3_value,clickData['points'][0])
        
        data = pd.read_json(jsonified_cleaned_data, orient='split')
        
        df=data[data[dropdown3_value].isin([clickData['points'][0]['text']])]
        df=df.dropna(subset=[dropdown1_value,dropdown2_value,'mktcap'])
        
        size=np.log(df['mktcap'].values)*3
        
        for i in range(len(size)): 
            if size[i]<0:
                size[i]=12

                  
        fig = px.scatter(df, x=dropdown1_value, y=dropdown2_value
                         ,size=size, color="ticker", text="ticker")
        
        model = sm.OLS(df[dropdown2_value], sm.add_constant(df[dropdown1_value])).fit()
        # print(model.summary())
        
        df['bestfit'] = sm.OLS(df[dropdown2_value], sm.add_constant(df[dropdown1_value])).fit().fittedvalues
        
        fig.add_trace(
            go.Scatter(x=df[dropdown1_value], y=df['bestfit'], name='trend',
                    line=dict(color='MediumPurple',width=2)))
        
        
        
        fig.update_layout(
            title = dropdown1_value+' vs '+ dropdown2_value,
                      xaxis_title = dropdown1_value,
                      yaxis_title = dropdown2_value
                      )
    
        return fig  

# =============================================================================
# Stk charts
# =============================================================================

@app.callback(Output('ts_plot3', 'figure'),
              [Input("scatter_plot2", "clickData"),
              Input('TS_data', 'data'),Input('myslider', 'value')])    

def update_timeseries3(clickData,jsonified_cleaned_data,myslider):
    # print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    df=data[data['ticker'] == ticker]
    
    dtemp=df[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11',  's_chg_12','s_chg_13','s_chg_14','s_chg_15','s_chg_16','s_chg_17','s_chg_18','s_chg_19'
          ,'s_chg_20','s_chg_21','s_chg_22','s_chg_23'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11',  'e_chg_12','e_chg_13','e_chg_14','e_chg_15','e_chg_16','e_chg_17','e_chg_18','e_chg_19'
          ,'e_chg_20','e_chg_21','e_chg_22','e_chg_23'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11',  'p_chg_12','p_chg_13','p_chg_14','p_chg_15','p_chg_16','p_chg_17','p_chg_18','p_chg_19'
          ,'p_chg_20','p_chg_21','p_chg_22','p_chg_23']]
    
      
    df_s1=dtemp[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7','s_chg_8','s_chg_9','s_chg_10','s_chg_11',
                 's_chg_12','s_chg_13','s_chg_14','s_chg_15','s_chg_16','s_chg_17','s_chg_18','s_chg_19','s_chg_20','s_chg_21','s_chg_22','s_chg_23']]
    df_s1=df_s1.rename({'s_chg_0': '0', 's_chg_1': '1','s_chg_2': '2',
                        's_chg_3': '3', 's_chg_4': '4','s_chg_5': '5',
                        's_chg_6': '6', 's_chg_7': '7', 's_chg_8': '8', 's_chg_9': '9'
                        , 's_chg_10': '10', 's_chg_11': '11',
                        's_chg_12': '12', 's_chg_13': '13','s_chg_14': '14',
                        's_chg_15': '15', 's_chg_16': '16','s_chg_17': '17',
                        's_chg_18': '18', 's_chg_19': '19', 's_chg_20': '20', 's_chg_21': '21'
                        , 's_chg_22': '22', 's_chg_23': '23'}
                        , axis=1)
    
    df_s2=dtemp[['e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7','e_chg_8','e_chg_9','e_chg_10','e_chg_11',
                 'e_chg_12','e_chg_13','e_chg_14','e_chg_15','e_chg_16','e_chg_17','e_chg_18','e_chg_19','e_chg_20','e_chg_21','e_chg_22','e_chg_23']]
    df_s2=df_s2.rename({'e_chg_0': '0', 'e_chg_1': '1','e_chg_2': '2',
                        'e_chg_3': '3', 'e_chg_4': '4','e_chg_5': '5',
                        'e_chg_6': '6', 'e_chg_7': '7', 'e_chg_8': '8', 'e_chg_9': '9'
                        , 'e_chg_10': '10', 'e_chg_11': '11',
                        'e_chg_12': '12', 'e_chg_13': '13','e_chg_14': '14',
                        'e_chg_15': '15', 'e_chg_16': '16','e_chg_17': '17',
                        'e_chg_18': '18', 'e_chg_19': '19', 'e_chg_20': '20', 'e_chg_21': '21'
                        , 'e_chg_22': '22', 'e_chg_23': '23'}
                        , axis=1)
    
    df_s3=dtemp[['p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7','p_chg_8','p_chg_9','p_chg_10','p_chg_11',
                 'p_chg_12','p_chg_13','p_chg_14','p_chg_15','p_chg_16','p_chg_17','p_chg_18','p_chg_19','p_chg_20','p_chg_21','p_chg_22','p_chg_23']]
    df_s3=df_s3.rename({'p_chg_0': '0', 'p_chg_1': '1','p_chg_2': '2',
                        'p_chg_3': '3', 'p_chg_4': '4','p_chg_5': '5',
                        'p_chg_6': '6', 'p_chg_7': '7', 'p_chg_8': '8', 'p_chg_9': '9'
                        , 'p_chg_10': '10', 'p_chg_11': '11',
                        'p_chg_12': '12', 'p_chg_13': '13','p_chg_14': '14',
                        'p_chg_15': '15', 'p_chg_16': '16','p_chg_17': '17',
                        'p_chg_18': '18', 'p_chg_19': '19', 'p_chg_20': '20', 'p_chg_21': '21'
                        , 'p_chg_22': '22', 'p_chg_23': '23'}
                        , axis=1)
    var=['s','e','p']
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    df1['period1'] = df1['period'].astype(int)
    df1 = df1[df1['period1'] <= myslider]
    df1=df1.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    df1['cum']= df1[['Rate']].cumsum()
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    df2['period1'] = df2['period'].astype(int)
    df2 = df2[df2['period1'] <= myslider]
    df2=df2.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    df2['cum']= df2[['Rate']].cumsum()
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3['period1'] = df3['period'].astype(int)
    df3 = df3[df3['period1'] <= myslider]
    df3=df3.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    df3['cum']= df3[['Rate']].cumsum()
    
    df = df1.append(df2)
    df = df.append(df3)
    # print(df.head())
       
    dmkt=data[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11',  's_chg_12','s_chg_13','s_chg_14','s_chg_15','s_chg_16','s_chg_17','s_chg_18','s_chg_19'
          ,'s_chg_20','s_chg_21','s_chg_22','s_chg_23'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11',  'e_chg_12','e_chg_13','e_chg_14','e_chg_15','e_chg_16','e_chg_17','e_chg_18','e_chg_19'
          ,'e_chg_20','e_chg_21','e_chg_22','e_chg_23'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11',  'p_chg_12','p_chg_13','p_chg_14','p_chg_15','p_chg_16','p_chg_17','p_chg_18','p_chg_19'
          ,'p_chg_20','p_chg_21','p_chg_22','p_chg_23']].mean()
    
    dmkt=dmkt.reset_index()
    
    dmkt.rename(columns={'index':'period',0:'Rate'}, inplace = True)

    dmkt_s=dmkt[dmkt['period'].isin(['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11',  's_chg_12','s_chg_13','s_chg_14','s_chg_15','s_chg_16','s_chg_17','s_chg_18','s_chg_19'
          ,'s_chg_20','s_chg_21','s_chg_22','s_chg_23'])]
    dmkt_s['period']=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'
                      , '16', '17', '18', '19', '20', '21', '22', '23']
    dmkt_s['period1'] = dmkt_s['period'].astype(int)
    dmkt_s = dmkt_s[dmkt_s['period1'] <= myslider]
    dmkt_s=dmkt_s.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    dmkt_s['cum']= dmkt_s[['Rate']].cumsum()    
    
    dmkt_e=dmkt[dmkt['period'].isin(['e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
   ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11',  'e_chg_12','e_chg_13','e_chg_14','e_chg_15','e_chg_16','e_chg_17','e_chg_18','e_chg_19'
      ,'e_chg_20','e_chg_21','e_chg_22','e_chg_23'])]
    dmkt_e['period']=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'
                      , '16', '17', '18', '19', '20', '21', '22', '23']
    dmkt_e['period1'] = dmkt_e['period'].astype(int)
    dmkt_e = dmkt_e[dmkt_e['period1'] <= myslider]
    dmkt_e=dmkt_e.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    dmkt_e['cum']= dmkt_e[['Rate']].cumsum()   
    
    dmkt_p=dmkt[dmkt['period'].isin(['p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
    ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11',  'p_chg_12','p_chg_13','p_chg_14','p_chg_15','p_chg_16','p_chg_17','p_chg_18','p_chg_19'
       ,'p_chg_20','p_chg_21','p_chg_22','p_chg_23'])]
    dmkt_p['period']=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'
                      , '16', '17', '18', '19', '20', '21', '22', '23']
    dmkt_p['period1'] = dmkt_p['period'].astype(int)
    dmkt_p = dmkt_p[dmkt_p['period1'] <= myslider]
    dmkt_p=dmkt_p.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    dmkt_p['cum']= dmkt_p[['Rate']].cumsum()       

    dmkt_s['indicator']=var[0]
    dmkt_e['indicator']=var[1]
    dmkt_p['indicator']=var[2]
        
        
    dmkt = dmkt_s.append(dmkt_e)
    dmkt = dmkt.append(dmkt_p)


    fig = make_subplots(rows=1, cols=1,specs=[[{"secondary_y": True}]])


    colors = ['red','blue','purple','darkcyan','darkred']

    for i in range(len(var)): 
        sig=var[i]
        
        df_temp=df[df['indicator'].isin([sig])]
        
        dmkt_temp=dmkt[dmkt['indicator'].isin([sig])]
        
        if i<2:     
            fig.add_trace(
                go.Scatter(x=df_temp.period,
                           y=df_temp.cum,
                           name=sig,
                            visible=True,
                            line=dict(width=1,color=colors[i]),
                        showlegend=True),row=1, col=1)
            
            fig.add_trace(
                go.Scatter(x=dmkt_temp.period,
                           y=dmkt_temp.cum,
                           name='mkt '+sig,
                            visible=True,
                            line=dict(width=1,color=colors[i],dash='dot'),
                        showlegend=True),row=1, col=1)
            
        else:
            fig.add_trace(
                go.Scatter(x=df_temp.period,
                           y=df_temp.cum,
                           name=sig,
                            visible=True,
                            line=dict(width=1,color=colors[i]),
                        showlegend=True),secondary_y=True,row=1, col=1)
            fig.add_trace(
                go.Scatter(x=dmkt_temp.period,
                           y=dmkt_temp.cum,
                           name='mkt '+sig,
                            visible=True,
                            line=dict(width=1,color=colors[i],dash='dot'),
                        showlegend=True),secondary_y=True,row=1, col=1)

    fig.update_layout(title = ticker+' trend',
                  xaxis_title = 'DATE',
                  yaxis_title = 'trend',
    paper_bgcolor="LightSteelBlue",
    legend=dict(orientation="h",
    yanchor="bottom",
    y=-0.2,
    xanchor="center",
    x=0.5))    
    
    return fig
       
###################

@app.callback(Output('ts_plot4', 'figure'),
              [Input("scatter_plot2", "clickData"), 
              Input('TS_data', 'data'),Input('myslider', 'value')])    

def update_timeseries4(clickData,jsonified_cleaned_data,myslider):
    
    # print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    df=data[data['ticker'] == ticker]
    
    dtemp=df[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11',  's_chg_12','s_chg_13','s_chg_14','s_chg_15','s_chg_16','s_chg_17','s_chg_18','s_chg_19'
          ,'s_chg_20','s_chg_21','s_chg_22','s_chg_23'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11',  'e_chg_12','e_chg_13','e_chg_14','e_chg_15','e_chg_16','e_chg_17','e_chg_18','e_chg_19'
          ,'e_chg_20','e_chg_21','e_chg_22','e_chg_23'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11',  'p_chg_12','p_chg_13','p_chg_14','p_chg_15','p_chg_16','p_chg_17','p_chg_18','p_chg_19'
          ,'p_chg_20','p_chg_21','p_chg_22','p_chg_23']]
    
      
    df_s1=dtemp[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7','s_chg_8','s_chg_9','s_chg_10','s_chg_11',
                 's_chg_12','s_chg_13','s_chg_14','s_chg_15','s_chg_16','s_chg_17','s_chg_18','s_chg_19','s_chg_20','s_chg_21','s_chg_22','s_chg_23']]
    df_s1=df_s1.rename({'s_chg_0': '0', 's_chg_1': '1','s_chg_2': '2',
                        's_chg_3': '3', 's_chg_4': '4','s_chg_5': '5',
                        's_chg_6': '6', 's_chg_7': '7', 's_chg_8': '8', 's_chg_9': '9'
                        , 's_chg_10': '10', 's_chg_11': '11',
                        's_chg_12': '12', 's_chg_13': '13','s_chg_14': '14',
                        's_chg_15': '15', 's_chg_16': '16','s_chg_17': '17',
                        's_chg_18': '18', 's_chg_19': '19', 's_chg_20': '20', 's_chg_21': '21'
                        , 's_chg_22': '22', 's_chg_23': '23'}
                        , axis=1)
    
    df_s2=dtemp[['e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7','e_chg_8','e_chg_9','e_chg_10','e_chg_11',
                 'e_chg_12','e_chg_13','e_chg_14','e_chg_15','e_chg_16','e_chg_17','e_chg_18','e_chg_19','e_chg_20','e_chg_21','e_chg_22','e_chg_23']]
    df_s2=df_s2.rename({'e_chg_0': '0', 'e_chg_1': '1','e_chg_2': '2',
                        'e_chg_3': '3', 'e_chg_4': '4','e_chg_5': '5',
                        'e_chg_6': '6', 'e_chg_7': '7', 'e_chg_8': '8', 'e_chg_9': '9'
                        , 'e_chg_10': '10', 'e_chg_11': '11',
                        'e_chg_12': '12', 'e_chg_13': '13','e_chg_14': '14',
                        'e_chg_15': '15', 'e_chg_16': '16','e_chg_17': '17',
                        'e_chg_18': '18', 'e_chg_19': '19', 'e_chg_20': '20', 'e_chg_21': '21'
                        , 'e_chg_22': '22', 'e_chg_23': '23'}
                        , axis=1)
    
    df_s3=dtemp[['p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7','p_chg_8','p_chg_9','p_chg_10','p_chg_11',
                 'p_chg_12','p_chg_13','p_chg_14','p_chg_15','p_chg_16','p_chg_17','p_chg_18','p_chg_19','p_chg_20','p_chg_21','p_chg_22','p_chg_23']]
    df_s3=df_s3.rename({'p_chg_0': '0', 'p_chg_1': '1','p_chg_2': '2',
                        'p_chg_3': '3', 'p_chg_4': '4','p_chg_5': '5',
                        'p_chg_6': '6', 'p_chg_7': '7', 'p_chg_8': '8', 'p_chg_9': '9'
                        , 'p_chg_10': '10', 'p_chg_11': '11',
                        'p_chg_12': '12', 'p_chg_13': '13','p_chg_14': '14',
                        'p_chg_15': '15', 'p_chg_16': '16','p_chg_17': '17',
                        'p_chg_18': '18', 'p_chg_19': '19', 'p_chg_20': '20', 'p_chg_21': '21'
                        , 'p_chg_22': '22', 'p_chg_23': '23'}
                        , axis=1)
    var=['s','e','p']
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    df1['period1'] = df1['period'].astype(int)
    df1 = df1[df1['period1'] <= myslider]
    df1=df1.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    df2['period1'] = df2['period'].astype(int)
    df2 = df2[df2['period1'] <= myslider]
    df2=df2.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3['period1'] = df3['period'].astype(int)
    df3 = df3[df3['period1'] <= myslider]
    df3=df3.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))

    df = df1.append(df2)
    df = df.append(df3)

    fig = make_subplots(rows=1, cols=1,specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df1.period, y=df1.Rate, fill='tozeroy',
                        name='s', mode='none' # override default markers+lines
                        ))
    fig.add_trace(go.Scatter(x=df3.period, y=df3.Rate, fill='tonexty',
                        name='p', mode= 'none'),secondary_y=True,)
    
    fig.update_layout(
    paper_bgcolor="LightSteelBlue",
    legend=dict(orientation="h",
    yanchor="bottom",
    y=-0.2,
    xanchor="center",
    x=0.5))
    
    return fig

################
# =============================================================================
# P trend t stat charts
# =============================================================================
@app.callback(Output('ts_plot5', 'figure'),
              [Input("scatter_plot2", "clickData"),
              Input('TS_data', 'data'),Input('myslider', 'value')])    

def update_timeseries3(clickData,jsonified_cleaned_data,myslider):
    # print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    df=data[data['ticker'] == ticker]
    
   
    df_s3=df[['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11','p_tst_12','p_tst_13','p_tst_14','p_tst_15'
       ,'p_tst_16','p_tst_17','p_tst_18','p_tst_19','p_tst_20','p_tst_21','p_tst_22','p_tst_23']]
    df_s3=df_s3.rename({'p_tst_0': '0', 'p_tst_1': '1','p_tst_2': '2',
                        'p_tst_3': '3', 'p_tst_4': '4','p_tst_5': '5',
                        'p_tst_6': '6', 'p_tst_7': '7', 'p_tst_8': '8', 'p_tst_9': '9'
                        , 'p_tst_10': '10', 'p_tst_11': '11',
                        'p_tst_12': '12', 'p_tst_13': '13','p_tst_14': '14',
                        'p_tst_15': '15', 'p_tst_16': '16','p_tst_17': '17',
                        'p_tst_18': '18', 'p_tst_19': '19', 'p_tst_20': '20', 'p_tst_21': '21'
                        , 'p_tst_22': '22', 'p_tst_23': '23'}
                        , axis=1)
   
    df3=df_s3
    df3['indicator']='p'
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3['period1'] = df3['period'].astype(int)
    df3 = df3[df3['period1'] <= myslider]
    df3=df3.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    df3['cum']= df3[['Rate']].cumsum()
    
       
    dmkt=data[['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11','p_tst_12','p_tst_13','p_tst_14','p_tst_15'
       ,'p_tst_16','p_tst_17','p_tst_18','p_tst_19','p_tst_20','p_tst_21','p_tst_22','p_tst_23']].median()
    
    dmkt=dmkt.reset_index()

    dmkt.rename(columns={'index':'period',0:'Rate'}, inplace = True)
    
    dmkt_p=dmkt[dmkt['period'].isin(['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11','p_tst_12','p_tst_13','p_tst_14','p_tst_15'
       ,'p_tst_16','p_tst_17','p_tst_18','p_tst_19','p_tst_20','p_tst_21','p_tst_22','p_tst_23'])]
    dmkt_p['period']=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'
                      , '16', '17', '18', '19', '20', '21', '22', '23']
    dmkt_p=dmkt_p.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    dmkt_p['period1'] = dmkt_p['period'].astype(int)
    dmkt_p = dmkt_p[dmkt_p['period1'] <= myslider]
    dmkt_p['cum']= dmkt_p[['Rate']].cumsum()       
    
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df3.period, y=df3.Rate, fill='tozeroy',
                        name=ticker+' p_tst40d', mode='none' # override default markers+lines
                        ))
    fig.add_trace(go.Scatter(x=dmkt_p.period, y=dmkt_p.Rate, fill='tonexty',
                        name='mkt p_tst40d', mode= 'none'))
    
    fig.update_layout(
    paper_bgcolor="LightSteelBlue",
    legend=dict(orientation="h",
    yanchor="bottom",
    y=-0.2,
    xanchor="center",
    x=0.5))
    
    return fig


######

@app.callback(Output('ts_plot6', 'figure'),
              [Input("scatter_plot2", "clickData"),
              Input('TS_data', 'data'),Input('myslider', 'value')])    

def update_timeseries(clickData,jsonified_cleaned_data,myslider):
    # print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    df=data[data['ticker'] == ticker]
    
     
    df_s3=df[['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11','p_tst_12','p_tst_13','p_tst_14','p_tst_15'
       ,'p_tst_16','p_tst_17','p_tst_18','p_tst_19','p_tst_20','p_tst_21','p_tst_22','p_tst_23']]
    df_s3=df_s3.rename({'p_tst_0': '0', 'p_tst_1': '1','p_tst_2': '2',
                        'p_tst_3': '3', 'p_tst_4': '4','p_tst_5': '5',
                        'p_tst_6': '6', 'p_tst_7': '7', 'p_tst_8': '8', 'p_tst_9': '9'
                        , 'p_tst_10': '10', 'p_tst_11': '11',
                        'p_tst_12': '12', 'p_tst_13': '13','p_tst_14': '14',
                        'p_tst_15': '15', 'p_tst_16': '16','p_tst_17': '17',
                        'p_tst_18': '18', 'p_tst_19': '19', 'p_tst_20': '20', 'p_tst_21': '21'
                        , 'p_tst_22': '22', 'p_tst_23': '23'}
                        , axis=1)
   
    df3=df_s3
    df3['indicator']='p'
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3['period1'] = df3['period'].astype(int)
    df3 = df3[df3['period1'] <= myslider]
    df3=df3.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    df3['cum']= df3[['Rate']].cumsum()
    
       
    dmkt=data[['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11','p_tst_12','p_tst_13','p_tst_14','p_tst_15'
       ,'p_tst_16','p_tst_17','p_tst_18','p_tst_19','p_tst_20','p_tst_21','p_tst_22','p_tst_23']].median()
    
    dmkt=dmkt.reset_index()

    dmkt.rename(columns={'index':'period',0:'Rate'}, inplace = True)
    
    dmkt_p=dmkt[dmkt['period'].isin(['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11','p_tst_12','p_tst_13','p_tst_14','p_tst_15'
       ,'p_tst_16','p_tst_17','p_tst_18','p_tst_19','p_tst_20','p_tst_21','p_tst_22','p_tst_23'])]
    dmkt_p['period']=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'
                      , '16', '17', '18', '19', '20', '21', '22', '23']
    dmkt_p['period1'] = dmkt_p['period'].astype(int)
    dmkt_p = dmkt_p[dmkt_p['period1'] <= myslider]
    dmkt_p=dmkt_p.sort_values('period', ascending=False, key=lambda s: s.str[0:].astype(int))
    dmkt_p['cum']= dmkt_p[['Rate']].cumsum()       
    
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df3.period, y=df3.cum, fill='tozeroy',
                        name=ticker+' cum p_tst', mode='none' # override default markers+lines
                        ))
    fig.add_trace(go.Scatter(x=dmkt_p.period, y=dmkt_p.cum, fill='tonexty',
                        name='mkt cum p_tst', mode= 'none'))
    
    fig.update_layout(
    paper_bgcolor="LightSteelBlue",
    legend=dict(orientation="h",
    yanchor="bottom",
    y=-0.2,
    xanchor="center",
    x=0.5))
    
    return fig
#########################

      
if __name__ == '__main__':
    app.run_server(debug=False)
    
os.kill(os.getpid(), signal.SIGTERM)









