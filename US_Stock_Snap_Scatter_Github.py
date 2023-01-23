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
        options = ['20230121','20221231', '20221130'],
        value = None, style={'width': '25%', 'display': 'inline-block'})
     
     ]),## drop down select y variable

    html.Div(children=[dcc.Slider(min=3, max=10, step=1, value=5, id='myslider')
    ]),


     html.Div([
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
    url_start = "https://github.com/xmanatsf/test/blob/main/US%20Screen%20monthly%20with%20universal%20variable1%20"
    url_end = ".xlsx?raw=true"
    query= url_start + dt1 + url_end
    df=pd.read_excel(query,engine='openpyxl')
   

    return df.to_json(date_format='iso', orient='split') 

    
    return df.to_json(date_format='iso', orient='split')    



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
        
        fig.add_trace(
            go.Scatter(mode='markers',x=df['x_var'], y=df['y_var'], name='all',
                        opacity=0.3,text = df['ind'],
                    hovertemplate=f"<b> ind: %{{text}}</b><br><b>X: %{{x}}<br><b>Y:</b> %{{y}}<extra></extra>",
                    marker=dict(color='gray',size=10)))
        
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
              ,Input('TS_data', 'data')])    

def update_timeseries1(clickData,dropdown3_value,jsonified_cleaned_data):
    
    # print(clickData['points'][0])
    
    ind=clickData['points'][0]['text']
    
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    
    df=data[data[dropdown3_value].isin([clickData['points'][0]['text']])]
    
    # print(df.head())
    
   
    dtemp=df[[dropdown3_value, 's_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11']]
    
 
    dtemp=dtemp.groupby([dropdown3_value]).agg(
        s0=('s_chg_0', 'mean'),s1=('s_chg_1', 'mean'),s2=('s_chg_2', 'mean'),s3=('s_chg_3', 'mean'),
        s4=('s_chg_4', 'mean'),s5=('s_chg_5', 'mean'),s6=('s_chg_6', 'mean'),s7=('s_chg_7', 'mean'),
        s8=('s_chg_8', 'mean'),s9=('s_chg_9', 'mean'),s10=('s_chg_10', 'mean'),s11=('s_chg_11', 'mean'),
        e0=('e_chg_0', 'mean'),e1=('e_chg_1', 'mean'),e2=('e_chg_2', 'mean'),e3=('e_chg_3', 'mean'),
        e4=('e_chg_4', 'mean'),e5=('e_chg_5', 'mean'),e6=('e_chg_6', 'mean'),e7=('e_chg_7', 'mean'),
        e8=('e_chg_8', 'mean'),e9=('e_chg_9', 'mean'),e10=('e_chg_10', 'mean'),e11=('e_chg_11', 'mean'),
        p0=('p_chg_0', 'mean'),p1=('p_chg_1', 'mean'),p2=('p_chg_2', 'mean'),p3=('p_chg_3', 'mean'),
        p4=('p_chg_4', 'mean'),p5=('p_chg_5', 'mean'),p6=('p_chg_6', 'mean'),p7=('p_chg_7', 'mean'),
        p8=('p_chg_8', 'mean'),p9=('p_chg_9', 'mean'),p10=('p_chg_10', 'mean'),p11=('p_chg_11', 'mean'))
    
    dtemp=dtemp[dtemp.index!='@NA']
    dtemp=dtemp.reset_index()
      
    df_s1=dtemp[['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11']]
    df_s1=df_s1.rename({'s0': 'm0', 's1': 'm1','s2': 'm2',
                        's3': 'm3', 's4': 'm4','s5': 'm5',
                        's6': 'm6', 's7': 'm7', 's8': 'm8', 's9': 'm9'
                        , 's10': 'm91', 's11': 'm92'}
                        , axis=1)
    
       
    df_s2=dtemp[['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11']]
    df_s2=df_s2.rename({'e0': 'm0', 'e1': 'm1','e2': 'm2',
                        'e3': 'm3', 'e4': 'm4','e5': 'm5',
                        'e6': 'm6', 'e7': 'm7', 'e8': 'm8', 'e9': 'm9'
                        , 'e10': 'm91', 'e11': 'm92'}
                        , axis=1)
    
    df_s3=dtemp[['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11']]
    df_s3=df_s3.rename({'p0': 'm0', 'p1': 'm1','p2': 'm2',
                        'p3': 'm3', 'p4': 'm4','p5': 'm5',
                        'p6': 'm6', 'p7': 'm7', 'p8': 'm8', 'p9': 'm9'
                        , 'p10': 'm91', 'p11': 'm92'}
                        , axis=1)
    var=['s','e','p']
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    df1=df1.sort_values('period', ascending=False)
    df1['cum']= df1[['Rate']].cumsum()
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    df2=df2.sort_values('period', ascending=False)
    df2['cum']= df2[['Rate']].cumsum()
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3=df3.sort_values('period', ascending=False)
    df3['cum']= df3[['Rate']].cumsum()
    
    df = df1.append(df2)
    df = df.append(df3)
    # print(df.head())
       
    dmkt=data[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11']].mean()
    
    dmkt=dmkt.reset_index()

    dmkt.rename(columns={'index':'period',0:'Rate'}, inplace = True)
    dmkt_s=dmkt[dmkt['period'].isin(['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11'])]
    dmkt_s['period']=['m0','m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm91', 'm92']
    dmkt_s=dmkt_s.sort_values('period', ascending=False)
    dmkt_s['cum']= dmkt_s[['Rate']].cumsum()    
    
    dmkt_e=dmkt[dmkt['period'].isin(['e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11'])]
    dmkt_e['period']=['m0','m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm91', 'm92']
    dmkt_e=dmkt_e.sort_values('period', ascending=False)
    dmkt_e['cum']= dmkt_e[['Rate']].cumsum()   
    
    dmkt_p=dmkt[dmkt['period'].isin(['p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11'])]
    dmkt_p['period']=['m0','m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm91', 'm92']
    dmkt_p=dmkt_p.sort_values('period', ascending=False)
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
              ,Input('TS_data', 'data')])    

def update_timeseries2(clickData,dropdown3_value,jsonified_cleaned_data):
    
    
    ind=clickData['points'][0]['text']
    
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    
    df=data[data[dropdown3_value].isin([clickData['points'][0]['text']])]
    
    # print(df.head())
    
   
    dtemp=df[[dropdown3_value, 's_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11']]
    
 
    dtemp=dtemp.groupby([dropdown3_value]).agg(
        s0=('s_chg_0', 'mean'),s1=('s_chg_1', 'mean'),s2=('s_chg_2', 'mean'),s3=('s_chg_3', 'mean'),
        s4=('s_chg_4', 'mean'),s5=('s_chg_5', 'mean'),s6=('s_chg_6', 'mean'),s7=('s_chg_7', 'mean'),
        s8=('s_chg_8', 'mean'),s9=('s_chg_9', 'mean'),s10=('s_chg_10', 'mean'),s11=('s_chg_11', 'mean'),
        e0=('e_chg_0', 'mean'),e1=('e_chg_1', 'mean'),e2=('e_chg_2', 'mean'),e3=('e_chg_3', 'mean'),
        e4=('e_chg_4', 'mean'),e5=('e_chg_5', 'mean'),e6=('e_chg_6', 'mean'),e7=('e_chg_7', 'mean'),
        e8=('e_chg_8', 'mean'),e9=('e_chg_9', 'mean'),e10=('e_chg_10', 'mean'),e11=('e_chg_11', 'mean'),
        p0=('p_chg_0', 'mean'),p1=('p_chg_1', 'mean'),p2=('p_chg_2', 'mean'),p3=('p_chg_3', 'mean'),
        p4=('p_chg_4', 'mean'),p5=('p_chg_5', 'mean'),p6=('p_chg_6', 'mean'),p7=('p_chg_7', 'mean'),
        p8=('p_chg_8', 'mean'),p9=('p_chg_9', 'mean'),p10=('p_chg_10', 'mean'),p11=('p_chg_11', 'mean'))
    
    dtemp=dtemp[dtemp.index!='@NA']
    dtemp=dtemp.reset_index()
      
    df_s1=dtemp[['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11']]
    df_s1=df_s1.rename({'s0': 'm0', 's1': 'm1','s2': 'm2',
                        's3': 'm3', 's4': 'm4','s5': 'm5',
                        's6': 'm6', 's7': 'm7', 's8': 'm8', 's9': 'm9'
                        , 's10': 'm91', 's11': 'm92'}
                        , axis=1)
    
    df_s2=dtemp[['e0','e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11']]
    df_s2=df_s2.rename({'e0': 'm0', 'e1': 'm1','e2': 'm2',
                        'e3': 'm3', 'e4': 'm4','e5': 'm5',
                        'e6': 'm6', 'e7': 'm7', 'e8': 'm8', 'e9': 'm9'
                        , 'e10': 'm91', 'e11': 'm92'}
                        , axis=1)
    
    df_s3=dtemp[['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11']]
    df_s3=df_s3.rename({'p0': 'm0', 'p1': 'm1','p2': 'm2',
                        'p3': 'm3', 'p4': 'm4','p5': 'm5',
                        'p6': 'm6', 'p7': 'm7', 'p8': 'm8', 'p9': 'm9'
                        , 'p10': 'm91', 'p11': 'm92'}
                        , axis=1)
    var=['s','e','p']
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    df1=df1.sort_values('period', ascending=False)
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    df2=df2.sort_values('period', ascending=False)
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3=df3.sort_values('period', ascending=False)

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
              Input('TS_data', 'data')])    

def update_timeseries3(clickData,jsonified_cleaned_data):
    # print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    df=data[data['ticker'] == ticker]
    
    dtemp=df[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11']]
    
      
    df_s1=dtemp[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11']]
    df_s1=df_s1.rename({'s_chg_0': 'm0', 's_chg_1': 'm1','s_chg_2': 'm2',
                        's_chg_3': 'm3', 's_chg_4': 'm4','s_chg_5': 'm5',
                        's_chg_6': 'm6', 's_chg_7': 'm7', 's_chg_8': 'm8', 's_chg_9': 'm9'
                        , 's_chg_10': 'm91', 's_chg_11': 'm92'}
                        , axis=1)
    
       
    df_s2=dtemp[['e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
   ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11']]
    df_s2=df_s2.rename({'e_chg_0': 'm0', 'e_chg_1': 'm1','e_chg_2': 'm2',
                        'e_chg_3': 'm3', 'e_chg_4': 'm4','e_chg_5': 'm5',
                        'e_chg_6': 'm6', 'e_chg_7': 'm7', 'e_chg_8': 'm8', 'e_chg_9': 'm9'
                        , 'e_chg_10': 'm91', 'e_chg_11': 'm92'}
                        , axis=1)
    
    df_s3=dtemp[['p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
    ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11']]
    df_s3=df_s3.rename({'p_chg_0': 'm0', 'p_chg_1': 'm1','p_chg_2': 'm2',
                        'p_chg_3': 'm3', 'p_chg_4': 'm4','p_chg_5': 'm5',
                        'p_chg_6': 'm6', 'p_chg_7': 'm7', 'p_chg_8': 'm8', 'p_chg_9': 'm9'
                        , 'p_chg_10': 'm91', 'p_chg_11': 'm92'}
                        , axis=1)
    var=['s','e','p']
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    df1=df1.sort_values('period', ascending=False)
    df1['cum']= df1[['Rate']].cumsum()
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    df2=df2.sort_values('period', ascending=False)
    df2['cum']= df2[['Rate']].cumsum()
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3=df3.sort_values('period', ascending=False)
    df3['cum']= df3[['Rate']].cumsum()
    
    df = df1.append(df2)
    df = df.append(df3)
    # print(df.head())
       
    dmkt=data[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11']].mean()
    
    dmkt=dmkt.reset_index()

    dmkt.rename(columns={'index':'period',0:'Rate'}, inplace = True)
    dmkt_s=dmkt[dmkt['period'].isin(['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11'])]
    dmkt_s['period']=['m0','m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm91', 'm92']
    dmkt_s=dmkt_s.sort_values('period', ascending=False)
    dmkt_s['cum']= dmkt_s[['Rate']].cumsum()    
    
    dmkt_e=dmkt[dmkt['period'].isin(['e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11'])]
    dmkt_e['period']=['m0','m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm91', 'm92']
    dmkt_e=dmkt_e.sort_values('period', ascending=False)
    dmkt_e['cum']= dmkt_e[['Rate']].cumsum()   
    
    dmkt_p=dmkt[dmkt['period'].isin(['p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11'])]
    dmkt_p['period']=['m0','m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm91', 'm92']
    dmkt_p=dmkt_p.sort_values('period', ascending=False)
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
              Input('TS_data', 'data')])    

def update_timeseries4(clickData,jsonified_cleaned_data):
    
    # print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    df=data[data['ticker'] == ticker]
    
    dtemp=df[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11'
        ,'e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
       ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11'
       ,'p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
       ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11']]
    
      
    df_s1=dtemp[['s_chg_0','s_chg_1','s_chg_2','s_chg_3','s_chg_4','s_chg_5','s_chg_6','s_chg_7'
       ,'s_chg_8','s_chg_9','s_chg_10','s_chg_11']]
    df_s1=df_s1.rename({'s_chg_0': 'm0', 's_chg_1': 'm1','s_chg_2': 'm2',
                        's_chg_3': 'm3', 's_chg_4': 'm4','s_chg_5': 'm5',
                        's_chg_6': 'm6', 's_chg_7': 'm7', 's_chg_8': 'm8', 's_chg_9': 'm9'
                        , 's_chg_10': 'm91', 's_chg_11': 'm92'}
                        , axis=1)
    
       
    df_s2=dtemp[['e_chg_0','e_chg_1','e_chg_2','e_chg_3','e_chg_4','e_chg_5','e_chg_6','e_chg_7'
   ,'e_chg_8','e_chg_9','e_chg_10','e_chg_11']]
    df_s2=df_s2.rename({'e_chg_0': 'm0', 'e_chg_1': 'm1','e_chg_2': 'm2',
                        'e_chg_3': 'm3', 'e_chg_4': 'm4','e_chg_5': 'm5',
                        'e_chg_6': 'm6', 'e_chg_7': 'm7', 'e_chg_8': 'm8', 'e_chg_9': 'm9'
                        , 'e_chg_10': 'm91', 'e_chg_11': 'm92'}
                        , axis=1)
    
    df_s3=dtemp[['p_chg_0','p_chg_1','p_chg_2','p_chg_3','p_chg_4','p_chg_5','p_chg_6','p_chg_7'
    ,'p_chg_8','p_chg_9','p_chg_10','p_chg_11']]
    df_s3=df_s3.rename({'p_chg_0': 'm0', 'p_chg_1': 'm1','p_chg_2': 'm2',
                        'p_chg_3': 'm3', 'p_chg_4': 'm4','p_chg_5': 'm5',
                        'p_chg_6': 'm6', 'p_chg_7': 'm7', 'p_chg_8': 'm8', 'p_chg_9': 'm9'
                        , 'p_chg_10': 'm91', 'p_chg_11': 'm92'}
                        , axis=1)
    var=['s','e','p']
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    df1=df1.sort_values('period', ascending=False)
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    df2=df2.sort_values('period', ascending=False)
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3=df3.sort_values('period', ascending=False)

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
              Input('TS_data', 'data')])    

def update_timeseries3(clickData,jsonified_cleaned_data):
    # print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    df=data[data['ticker'] == ticker]
    
    dtemp=df[['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11']]
    
      
    df_s3=dtemp[['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11']]
    df_s3=df_s3.rename({'p_tst_0': 'm0', 'p_tst_1': 'm1','p_tst_2': 'm2',
                        'p_tst_3': 'm3', 'p_tst_4': 'm4','p_tst_5': 'm5',
                        'p_tst_6': 'm6', 'p_tst_7': 'm7', 'p_tst_8': 'm8', 'p_tst_9': 'm9'
                        , 'p_tst_10': 'm91', 'p_tst_11': 'm92'}
                        , axis=1)
   
    df3=df_s3
    df3['indicator']='p'
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3=df3.sort_values('period', ascending=False)
    df3['cum']= df3[['Rate']].cumsum()
    
       
    dmkt=data[['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11']].median()
    
    dmkt=dmkt.reset_index()

    dmkt.rename(columns={'index':'period',0:'Rate'}, inplace = True)
    
    dmkt_p=dmkt[dmkt['period'].isin(['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11'])]
    dmkt_p['period']=['m0','m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm91', 'm92']
    dmkt_p=dmkt_p.sort_values('period', ascending=False)
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
              Input('TS_data', 'data')])    

def update_timeseries(clickData,jsonified_cleaned_data):
    # print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    df=data[data['ticker'] == ticker]
    
    dtemp=df[['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11']]
    
      
    df_s3=dtemp[['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11']]
    df_s3=df_s3.rename({'p_tst_0': 'm0', 'p_tst_1': 'm1','p_tst_2': 'm2',
                        'p_tst_3': 'm3', 'p_tst_4': 'm4','p_tst_5': 'm5',
                        'p_tst_6': 'm6', 'p_tst_7': 'm7', 'p_tst_8': 'm8', 'p_tst_9': 'm9'
                        , 'p_tst_10': 'm91', 'p_tst_11': 'm92'}
                        , axis=1)
   
    df3=df_s3
    df3['indicator']='p'
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    df3=df3.sort_values('period', ascending=False)
    df3['cum']= df3[['Rate']].cumsum()
    
       
    dmkt=data[['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11']].median()
    
    dmkt=dmkt.reset_index()

    dmkt.rename(columns={'index':'period',0:'Rate'}, inplace = True)
    
    dmkt_p=dmkt[dmkt['period'].isin(['p_tst_0','p_tst_1','p_tst_2','p_tst_3','p_tst_4','p_tst_5','p_tst_6','p_tst_7'
       ,'p_tst_8','p_tst_9','p_tst_10','p_tst_11'])]
    dmkt_p['period']=['m0','m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm91', 'm92']
    dmkt_p=dmkt_p.sort_values('period', ascending=False)
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









