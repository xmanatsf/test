import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

app = Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

app.layout = html.Div([
    html.H1(id = 'H1', children = 'factor scatter', style = {'textAlign':'center',\
                                            'marginTop':40,'marginBottom':40}),## Title line
     html.Div(children=[
         dcc.Dropdown( id = 'dropdown1',
        options = ['p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y'
                   ,'p_chg6m','p_chg3m','p_chg400d','p_chg300d','p_chg200d','p_chg150d'
                   ,'p_chg120d','p_chg70d','p_chg40d','p_chg20d','p_tstat_10y','p_tstat_7y'
                   ,'p_tstat_5y','p_tstat_3y','p_tstat_2y','p_tstat_1y','p_tstat_6m'
                   ,'p_tstat_3m','p_tstat_1m','p_tstat_200d','p_tstat_120d','p_tstat_50d'],
        value = None, style={'width': '25%', 'display': 'inline-block'}),## drop down select x variable
     
     dcc.Dropdown( id = 'dropdown2',
        options = ['p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y'
                   ,'p_chg6m','p_chg3m','p_chg400d','p_chg300d','p_chg200d','p_chg150d'
                   ,'p_chg120d','p_chg70d','p_chg40d','p_chg20d','p_tstat_10y','p_tstat_7y'
                   ,'p_tstat_5y','p_tstat_3y','p_tstat_2y','p_tstat_1y','p_tstat_6m'
                   ,'p_tstat_3m','p_tstat_1m','p_tstat_200d','p_tstat_120d','p_tstat_50d'],
        value = None, style={'width': '25%', 'display': 'inline-block'}),
     
     dcc.Dropdown( id = 'dropdown3',
        options = ['SEC', 'INDG', 'IND', 'RBICS_SEC','RBICS_SEC1','RBICS_SEC2'],
        value = None, style={'width': '25%', 'display': 'inline-block'}),
     
     dcc.Dropdown( id = 'dropdown4',
        options = ['20230103', '20221123', '20221001'],
        value = None, style={'width': '25%', 'display': 'inline-block'})
     
     ]),## drop down select y variable

     html.Div([
         dcc.Graph(id = 'scatter_plot1',clickData={'points': [{'text': 'nvda'}]}
          ,style={'width': '98%','height': '400px','display': 'inline-block'}),  #hoverData={'points': [{'ticker': 'nvda'}]}  sector specific scatter plot
         ]),

    html.Div(children=[
        dcc.Graph(id = 'ts_plot5',clickData={'points': [{'text': 'nvda'}]}
         ,style={'width': '49%','display': 'inline-block'}),  #hoverData={'points': [{'ticker': 'nvda'}]}  sector specific scatter plot
        dcc.Graph(id = 'ts_plot6',clickData={'points': [{'text': 'nvda'}]}
     ,style={'width': '49%','display': 'inline-block'}), 
    ]),
     
     html.Div([
         dcc.Graph(id = 'scatter_plot2',clickData={'points': [{'text': 'nvda'}]}
          ,style={'width': '98%','height': '400px','display': 'inline-block'}),  #hoverData={'points': [{'ticker': 'nvda'}]}  sector specific scatter plot
         ]),
   

    
    html.Div(children=[
        dcc.Graph(id = 'ts_plot1',clickData={'points': [{'text': 'nvda'}]}
         ,style={'width': '49%','display': 'inline-block'}),  #hoverData={'points': [{'ticker': 'nvda'}]}  sector specific scatter plot
    
        dcc.Graph(id = 'ts_plot2',clickData={'points': [{'text': 'nvda'}]}
         ,style={'width': '49%','display': 'inline-block'}) ## stock specific line plot
    ]),
    
    html.Div(children=[
        dcc.Graph(id = 'ts_plot3',clickData={'points': [{'text': 'nvda'}]}
         ,style={'width': '49%','display': 'inline-block'}),  #hoverData={'points': [{'ticker': 'nvda'}]}  sector specific scatter plot
    
        dcc.Graph(id = 'ts_plot4',clickData={'points': [{'text': 'nvda'}]}
         ,style={'width': '49%','display': 'inline-block'}) ## stock specific line plot
    ]),
    
    html.Div(children=[
        dcc.Graph(id = 'bar_chart1',clickData={'points': [{'text': 'nvda'}]}
         ,style={'width': '49%','display': 'inline-block'}),  #hoverData={'points': [{'ticker': 'nvda'}]}  sector specific scatter plot
    
        dcc.Graph(id = 'bar_chart2',clickData={'points': [{'text': 'nvda'}]}
         ,style={'width': '49%','display': 'inline-block'}) ## stock specific line plot
    ]),
    
    # dcc.Store stores the intermediate value
    dcc.Store(id='Scatter_data')
    
])


# =============================================================================
# store fetched data in the odyssey_data     
# =============================================================================

@app.callback(Output('Scatter_data', 'data'),
              Input("dropdown4", "value"))
def odyssey_data(dropdown4_value):
     # some expensive data processing step
    print(dropdown4_value)
    
    dt1=dropdown4_value
    url_start = "https://github.com/xmanatsf/test/blob/main/RIY%20AND%20SP1500%20MONITORING%20SCREEN%20UPDATE4%20"
    url_end = ".xlsx?raw=true"
    query= url_start + dt1 + url_end
    df=pd.read_excel(query,engine='openpyxl')
    
    print(df.head())
    
     # more generally, this line would be
     # json.dumps(cleaned_df)
    return df.to_json(date_format='iso', orient='split')    

# =============================================================================
# add sector level scatter plot    
# =============================================================================
    
@app.callback(Output('scatter_plot1', 'figure'),
              [Input("dropdown1", "value"), Input("dropdown2", "value")
               , Input("dropdown3", "value"),Input('Scatter_data', 'data')])    

def update_chart1(dropdown1_value,dropdown2_value,dropdown3_value,jsonified_cleaned_data):
        print(dropdown1_value,dropdown2_value,dropdown3_value)
        
        data = pd.read_json(jsonified_cleaned_data, orient='split')
        
        df=data[[dropdown3_value, 'ticker',dropdown1_value,dropdown2_value,'mktcap']]
        
        print(df.head())
        
        df1=df.groupby([dropdown3_value]).agg(
            x_var=(dropdown1_value, 'mean'),
            y_var=(dropdown2_value, 'mean'), 
            mktcap=('mktcap', 'mean')
            )
        
        df=df1[df1.index!='@NA']
        df=df.reset_index()
        
        print(df.head())
        
        df=df.dropna(how='all')
        df['ind']=df[dropdown3_value]
        
        
        
        size=np.log(df['mktcap'].values)*3
        fig = px.scatter(df, x='x_var', y='y_var'
                         ,size=size, color='ind', text='ind')
        
        model = sm.OLS(df['y_var'], sm.add_constant(df['x_var'])).fit()
        print(model.summary())
        
        df['bestfit'] = sm.OLS(df['y_var'], sm.add_constant(df['x_var'])).fit().fittedvalues
        
        fig.add_trace(
            go.Scatter(x=df['x_var'], y=df['bestfit'], name='trend',
                    line=dict(color='MediumPurple',width=2)))
        
        fig.add_trace(
            go.Scatter(mode='markers',x=df['x_var'], y=df['y_var'], name='all',
                        opacity=0.3,text = df['ind'],
                    hovertemplate=f"<b> ind: %{{text}}</b><br><b>X: %{{x}}<br><b>Y:</b> %{{y}}<extra></extra>",
                    marker=dict(color='gray',size=10)))
        
        
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
# end    
# =============================================================================

# =============================================================================
# Ind charts
# =============================================================================

@app.callback(Output('ts_plot5', 'figure'),
              [Input("scatter_plot1", "clickData"), Input("dropdown3", "value")
               ,Input('Scatter_data', 'data')])    

def update_timeseries5(clickData,dropdown3_value,jsonified_cleaned_data):
    
    print(clickData['points'][0])
    
    ind=clickData['points'][0]['text']
    
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    
    df=data[data[dropdown3_value].isin([clickData['points'][0]['text']])]
    
    dtemp=df[['p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y','p_chg6m','p_chg3m'
                , 's_chg10y','s_chg7y','s_chg5y','s_chg3y','s_chg2y','s_chg1y','s_chg6m','s_chg3m'
              ,'e_chg10y','e_chg7y','e_chg5y','e_chg3y','e_chg2y','e_chg1y','e_chg6m','e_chg3m']]
    
    dtemp1=data[[dropdown3_value]]

    dtemp=dtemp.join(dtemp1)

    dtemp=dtemp.groupby([dropdown3_value]).agg(
        p_chg10y=('p_chg10y', 'mean'), p_chg7y=('p_chg7y', 'mean'), 
        p_chg5y=('p_chg5y', 'mean'), p_chg3y=('p_chg3y', 'mean'), 
        p_chg2y=('p_chg2y', 'mean'), p_chg1y=('p_chg1y', 'mean'), 
        p_chg6m=('p_chg6m', 'mean'), p_chg3m=('p_chg3m', 'mean'),
        s_chg10y=('s_chg10y', 'mean'), s_chg7y=('s_chg7y', 'mean'), 
        s_chg5y=('s_chg5y', 'mean'), s_chg3y=('s_chg3y', 'mean'), 
        s_chg2y=('s_chg2y', 'mean'), s_chg1y=('s_chg1y', 'mean'), 
        s_chg6m=('s_chg6m', 'mean'), s_chg3m=('s_chg3m', 'mean'),
       e_chg10y=('e_chg10y', 'mean'), e_chg7y=('e_chg7y', 'mean'), 
        e_chg5y=('e_chg5y', 'mean'), e_chg3y=('e_chg3y', 'mean'), 
        e_chg2y=('e_chg2y', 'mean'), e_chg1y=('e_chg1y', 'mean'), 
        e_chg6m=('e_chg6m', 'mean'), e_chg3m=('e_chg3m', 'mean')
     )
    
    dtemp=dtemp[dtemp.index!='@NA']
    dtemp=dtemp.reset_index()

   
    
    df_s1=dtemp[['s_chg10y','s_chg7y','s_chg5y','s_chg3y','s_chg2y','s_chg1y','s_chg6m','s_chg3m']]
    df_s1=df_s1.rename({'s_chg10y': 'chg10y', 's_chg7y': 'chg7y','s_chg5y': 'chg5y',
                        's_chg3y': 'chg3y', 's_chg2y': 'chg2y','s_chg1y': 'chg1y',
                        's_chg6m': 'chg6m', 's_chg3m': 'chg3m'}
                        , axis=1)
    
    
    df_s2=dtemp[['e_chg10y','e_chg7y','e_chg5y','e_chg3y','e_chg2y','e_chg1y','e_chg6m','e_chg3m']]
    df_s2=df_s2.rename({'e_chg10y': 'chg10y', 'e_chg7y': 'chg7y','e_chg5y': 'chg5y',
                        'e_chg3y': 'chg3y', 'e_chg2y': 'chg2y','e_chg1y': 'chg1y',
                        'e_chg6m': 'chg6m', 'e_chg3m': 'chg3m'}
                        , axis=1)
     
    df_s3=dtemp[['p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y','p_chg6m','p_chg3m']]
    df_s3=df_s3.rename({'p_chg10y': 'chg10y', 'p_chg7y': 'chg7y','p_chg5y': 'chg5y',
                        'p_chg3y': 'chg3y', 'p_chg2y': 'chg2y','p_chg1y': 'chg1y',
                        'p_chg6m': 'chg6m', 'p_chg3m': 'chg3m'}
                        , axis=1)
   
    var=['s','e','p']
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')

    df = df1.append(df2)
    df = df.append(df3)
    
    print(df.head())
    
    fig = make_subplots(rows=1, cols=1,specs=[[{"secondary_y": True}]])


    colors = ['red','blue','purple','darkcyan','darkred']

    for i in range(len(var)): 
        sig=var[i]
        
        df_temp=df[df['indicator'].isin([sig])]
        
        if i<2:     
            fig.add_trace(
                go.Scatter(x=df_temp.period,
                           y=df_temp.Rate,
                           name=sig,
                            visible=True,
                            line=dict(width=1,color=colors[i]),
                        showlegend=True),row=1, col=1)
        else:
            fig.add_trace(
                go.Scatter(x=df_temp.period,
                           y=df_temp.Rate,
                           name=sig,
                            visible=True,
                            line=dict(width=1,color=colors[i]),
                        showlegend=True),secondary_y=True,row=1, col=1)

    fig.update_layout(title = ind+' absolute s e and p (RHS) trend',
                  xaxis_title = 'period',
                  yaxis_title = 'trend'
                  )                   
    
    return fig
       



@app.callback(Output('ts_plot6', 'figure'),
              [Input("scatter_plot1", "clickData"), Input("dropdown3", "value")
               ,Input('Scatter_data', 'data')])    

def update_timeseries6(clickData,dropdown3_value,jsonified_cleaned_data):
    
    data = pd.read_json(jsonified_cleaned_data, orient='split')
           
    dtemp=data[['p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y','p_chg6m','p_chg3m'
                , 's_chg10y','s_chg7y','s_chg5y','s_chg3y','s_chg2y','s_chg1y','s_chg6m','s_chg3m'
              ,'e_chg10y','e_chg7y','e_chg5y','e_chg3y','e_chg2y','e_chg1y','e_chg6m','e_chg3m']].rank(pct=True)
        
    dtemp1=data[[dropdown3_value]]

    dtemp=dtemp.join(dtemp1)
    
    ind=clickData['points'][0]['text']

    dtemp=dtemp[dtemp[dropdown3_value].isin([clickData['points'][0]['text']])]

    dtemp=dtemp.groupby([dropdown3_value]).agg(
        p_chg10y=('p_chg10y', 'mean'), p_chg7y=('p_chg7y', 'mean'), 
        p_chg5y=('p_chg5y', 'mean'), p_chg3y=('p_chg3y', 'mean'), 
        p_chg2y=('p_chg2y', 'mean'), p_chg1y=('p_chg1y', 'mean'), 
        p_chg6m=('p_chg6m', 'mean'), p_chg3m=('p_chg3m', 'mean'),
        s_chg10y=('s_chg10y', 'mean'), s_chg7y=('s_chg7y', 'mean'), 
        s_chg5y=('s_chg5y', 'mean'), s_chg3y=('s_chg3y', 'mean'), 
        s_chg2y=('s_chg2y', 'mean'), s_chg1y=('s_chg1y', 'mean'), 
        s_chg6m=('s_chg6m', 'mean'), s_chg3m=('s_chg3m', 'mean'),
       e_chg10y=('e_chg10y', 'mean'), e_chg7y=('e_chg7y', 'mean'), 
        e_chg5y=('e_chg5y', 'mean'), e_chg3y=('e_chg3y', 'mean'), 
        e_chg2y=('e_chg2y', 'mean'), e_chg1y=('e_chg1y', 'mean'), 
        e_chg6m=('e_chg6m', 'mean'), e_chg3m=('e_chg3m', 'mean')
     )
    
    dtemp=dtemp[dtemp.index!='@NA']
    dtemp=dtemp.reset_index()
    
  
    df_s1=dtemp[['s_chg10y','s_chg7y','s_chg5y','s_chg3y','s_chg2y','s_chg1y','s_chg6m','s_chg3m']]
    df_s1=df_s1.rename({'s_chg10y': 'chg10y', 's_chg7y': 'chg7y','s_chg5y': 'chg5y',
                        's_chg3y': 'chg3y', 's_chg2y': 'chg2y','s_chg1y': 'chg1y',
                        's_chg6m': 'chg6m', 's_chg3m': 'chg3m'}
                        , axis=1)

    df_s2=dtemp[['e_chg10y','e_chg7y','e_chg5y','e_chg3y','e_chg2y','e_chg1y','e_chg6m','e_chg3m']]
    df_s2=df_s2.rename({'e_chg10y': 'chg10y', 'e_chg7y': 'chg7y','e_chg5y': 'chg5y',
                        'e_chg3y': 'chg3y', 'e_chg2y': 'chg2y','e_chg1y': 'chg1y',
                        'e_chg6m': 'chg6m', 'e_chg3m': 'chg3m'}
                        , axis=1)
    
    df_s3=dtemp[['p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y','p_chg6m','p_chg3m']]
    df_s3=df_s3.rename({'p_chg10y': 'chg10y', 'p_chg7y': 'chg7y','p_chg5y': 'chg5y',
                        'p_chg3y': 'chg3y', 'p_chg2y': 'chg2y','p_chg1y': 'chg1y',
                        'p_chg6m': 'chg6m', 'p_chg3m': 'chg3m'}
                        , axis=1)
    
    var=['s','e','p']   
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    
    df = df1.append(df2)
    df = df.append(df3)
    
    print(df.head())
    
    fig = make_subplots(rows=1, cols=1)


    colors = ['red','blue','purple','darkcyan','darkred']

    for i in range(len(var)): 
        sig=var[i]
        
        df_temp=df[df['indicator'].isin([sig])]
             
        fig.add_trace(
            go.Scatter(x=df_temp.period,
                        y=df_temp.Rate,
                        name=sig,
                        visible=True,
                        line=dict(width=1,color=colors[i]),
                    showlegend=True),row=1, col=1)

    fig.update_layout(title = ind+' relative s e and p trend',
                  xaxis_title = 'period',
                  yaxis_title = 'trend'
                  )    
    return fig


# =============================================================================
# 
# =============================================================================
@app.callback(Output('scatter_plot2', 'figure'),
              [Input("dropdown1", "value"), Input("dropdown2", "value")
               , Input("dropdown3", "value"),Input("scatter_plot1", "clickData")
               ,Input('Scatter_data', 'data')])    

def update_chart2(dropdown1_value,dropdown2_value,dropdown3_value,clickData,jsonified_cleaned_data):
        print(dropdown1_value,dropdown2_value,dropdown3_value,clickData['points'][0])
        
        data = pd.read_json(jsonified_cleaned_data, orient='split')
        
        df=data[data[dropdown3_value].isin([clickData['points'][0]['text']])]
        df=df.dropna(subset=[dropdown1_value,dropdown2_value,'mktcap'])
        
        size=np.log(df['mktcap'].values)*3
        fig = px.scatter(df, x=dropdown1_value, y=dropdown2_value
                         ,size=size, color="ticker", text="ticker")
        
        model = sm.OLS(df[dropdown2_value], sm.add_constant(df[dropdown1_value])).fit()
        print(model.summary())
        
        df['bestfit'] = sm.OLS(df[dropdown2_value], sm.add_constant(df[dropdown1_value])).fit().fittedvalues
        
        fig.add_trace(
            go.Scatter(x=df[dropdown1_value], y=df['bestfit'], name='trend',
                    line=dict(color='MediumPurple',width=2)))
        
        # fig.add_trace(
        #     go.Scatter(mode='markers',x=data[dropdown1_value], y=data[dropdown2_value], name='all',
        #                opacity=0.3,text = data['ticker'],
        #             hovertemplate=f"<b> ticker: %{{text}}</b><br><b>X: %{{x}}<br><b>Y:</b> %{{y}}<extra></extra>",
        #             marker=dict(color='gray',size=10)))
        
        
        fig.update_layout(
            title = dropdown1_value+' vs '+ dropdown2_value,
                      xaxis_title = dropdown1_value,
                      yaxis_title = dropdown2_value
                      )
    
        return fig  




@app.callback(Output('ts_plot1', 'figure'),
              [Input("scatter_plot2", "clickData"),Input('Scatter_data', 'data')])    



def update_timeseries1(clickData,jsonified_cleaned_data):
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    
    # print(clickData['points'][0]['text'])
    print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    dff = data[data['ticker'] == ticker]
    print(dff.head())
    
    dtemp1=dff[['p_chg400d','p_chg300d','p_chg200d','p_chg150d','p_chg120d','p_chg70d','p_chg40d','p_chg20d'
                , 's_chg400d','s_chg300d','s_chg200d','s_chg150d','s_chg120d','s_chg70d','s_chg40d','s_chg20d'
              ,'e_chg400d','e_chg300d','e_chg200d','e_chg150d','e_chg120d','e_chg70d','e_chg40d','e_chg20d']]

    dtemp2=dff[['ticker']]
    dtemp=dtemp2.join(dtemp1)
    dtemp=dtemp.set_index('ticker')
    
    df_s1=dtemp[['s_chg400d','s_chg300d','s_chg200d','s_chg150d','s_chg120d','s_chg70d','s_chg40d','s_chg20d']]
    df_s1=df_s1.rename({'s_chg400d': 'chg400d', 's_chg300d': 'chg300d','s_chg200d': 'chg200d',
                        's_chg150d': 'chg150d', 's_chg120d': 'chg120d','s_chg70d': 'chg70d',
                        's_chg40d': 'chg40d', 's_chg20d': 'chg20d'}
                        , axis=1)
    
    
    df_s2=dtemp[['e_chg400d','e_chg300d','e_chg200d','e_chg150d','e_chg120d','e_chg70d','e_chg40d','e_chg20d']]
    df_s2=df_s2.rename({'e_chg400d': 'chg400d', 'e_chg300d': 'chg300d','e_chg200d': 'chg200d',
                        'e_chg150d': 'chg150d', 'e_chg120d': 'chg120d','e_chg70d': 'chg70d',
                        'e_chg40d': 'chg40d', 'e_chg20d': 'chg20d'}
                        , axis=1)
     
    df_s3=dtemp[['p_chg400d','p_chg300d','p_chg200d','p_chg150d','p_chg120d','p_chg70d','p_chg40d','p_chg20d']]
    df_s3=df_s3.rename({'p_chg400d': 'chg400d', 'p_chg300d': 'chg300d','p_chg200d': 'chg200d',
                        'p_chg150d': 'chg150d', 'p_chg120d': 'chg120d','p_chg70d': 'chg70d',
                        'p_chg40d': 'chg40d', 'p_chg20d': 'chg20d'}
                        , axis=1)
   
    var=['s','e','p']
    
    df1=df_s1
    df1['stk']=df_s1.index
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['stk','indicator'],var_name='period',value_name='Rate')
    
    df2=df_s2
    df2['stk']=df_s2.index
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['stk','indicator'],var_name='period',value_name='Rate')
    
    df3=df_s3
    df3['stk']=df_s3.index
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['stk','indicator'],var_name='period',value_name='Rate')

    df = df1.append(df2)
    df = df.append(df3)
    
    print(df.head())
    
    fig = make_subplots(rows=1, cols=1,specs=[[{"secondary_y": True}]])


    colors = ['red','blue','purple','darkcyan','darkred']

    for i in range(len(var)): 
        sig=var[i]
        
        df_temp=df[df['indicator'].isin([sig])]
        
        if i<2:     
            fig.add_trace(
                go.Scatter(x=df_temp.period,
                           y=df_temp.Rate,
                           name=sig,
                            visible=True,
                            line=dict(width=1,color=colors[i]),
                        showlegend=True),row=1, col=1)
        else:
            fig.add_trace(
                go.Scatter(x=df_temp.period,
                           y=df_temp.Rate,
                           name=sig,
                            visible=True,
                            line=dict(width=1,color=colors[i]),
                        showlegend=True),secondary_y=True,row=1, col=1)
             
        fig.update_layout(title = ticker+' absolute s e and p (RHS) trend',
                      xaxis_title = 'period',
                      yaxis_title = 'trend'
                      )
    return fig
       



@app.callback(Output('ts_plot2', 'figure'),
              [Input("scatter_plot2", "clickData"),Input('Scatter_data', 'data')])    



def update_timeseries2(clickData,jsonified_cleaned_data):
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    
    # print(clickData['points'][0]['text'])
    print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    dff = data[data['ticker'] == ticker]
    print(dff.head())
    
    
    dtemp1_rk=data[['p_chg400d','p_chg300d','p_chg200d','p_chg150d','p_chg120d','p_chg70d','p_chg40d','p_chg20d'
            , 's_chg400d','s_chg300d','s_chg200d','s_chg150d','s_chg120d','s_chg70d','s_chg40d','s_chg20d'
          ,'e_chg400d','e_chg300d','e_chg200d','e_chg150d','e_chg120d','e_chg70d','e_chg40d','e_chg20d']].rank(pct=True)
    dtemp2=data[['ticker']]
    dtemp_rk=dtemp2.join(dtemp1_rk)
    dtemp_rk = dtemp_rk[dtemp_rk['ticker'] == ticker]
    dtemp_rk=dtemp_rk.set_index('ticker')

    
    df_s1=dtemp_rk[['s_chg400d','s_chg300d','s_chg200d','s_chg150d','s_chg120d','s_chg70d','s_chg40d','s_chg20d']]
    df_s1=df_s1.rename({'s_chg400d': 'chg400d_rk', 's_chg300d': 'chg300d_rk','s_chg200d': 'chg200d_rk',
                        's_chg150d': 'chg150d_rk', 's_chg120d': 'chg120d_rk','s_chg70d': 'chg70d_rk',
                        's_chg40d': 'chg40d_rk', 's_chg20d': 'chg20d_rk'}
                        , axis=1)

    df_s2=dtemp_rk[['e_chg400d','e_chg300d','e_chg200d','e_chg150d','e_chg120d','e_chg70d','e_chg40d','e_chg20d']]
    df_s2=df_s2.rename({'e_chg400d': 'chg400d_rk', 'e_chg300d': 'chg300d_rk','e_chg200d': 'chg200d_rk',
                        'e_chg150d': 'chg150d_rk', 'e_chg120d': 'chg120d_rk','e_chg70d': 'chg70d_rk',
                        'e_chg40d': 'chg40d_rk', 'e_chg20d': 'chg20d_rk'}
                        , axis=1)
    
    df_s3=dtemp_rk[['p_chg400d','p_chg300d','p_chg200d','p_chg150d','p_chg120d','p_chg70d','p_chg40d','p_chg20d']]
    df_s3=df_s3.rename({'p_chg400d': 'chg400d_rk', 'p_chg300d': 'chg300d_rk','p_chg200d': 'chg200d_rk',
                        'p_chg150d': 'chg150d_rk', 'p_chg120d': 'chg120d_rk','p_chg70d': 'chg70d_rk',
                        'p_chg40d': 'chg40d_rk', 'p_chg20d': 'chg20d_rk'}
                        , axis=1)
    
    var=['s_rk','e_rk','p_rk']   
    
    df1=df_s1
    df1['stk']=df_s1.index
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['stk','indicator'],var_name='period',value_name='Rate')
    
    df2=df_s2
    df2['stk']=df_s2.index
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['stk','indicator'],var_name='period',value_name='Rate')
    
    df3=df_s3
    df3['stk']=df_s3.index
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['stk','indicator'],var_name='period',value_name='Rate')
    
    
    df_rk = df1.append(df2)
    df_rk = df_rk.append(df3)
        
    print(df_rk.head())
    
    fig = make_subplots(rows=1, cols=1)


    colors = ['red','blue','purple','darkcyan','darkred']

    for i in range(len(var)): 
        sig=var[i]
        
        df_temp=df_rk[df_rk['indicator'].isin([sig])]
             
        fig.add_trace(
            go.Scatter(x=df_temp.period,
                       y=df_temp.Rate,
                       name=sig,
                        visible=True,
                        line=dict(width=1,color=colors[i]),
                    showlegend=True),row=1, col=1)
    fig.update_layout(title = ticker+' relative s e and p trend',
                  xaxis_title = 'period',
                  yaxis_title = 'trend'
                  )
    return fig
      

@app.callback(Output('ts_plot3', 'figure'),
              [Input("scatter_plot2", "clickData"),Input('Scatter_data', 'data')])    

def update_timeseries3(clickData,jsonified_cleaned_data):
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    
    # print(clickData['points'][0]['text'])
    print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    dff = data[data['ticker'] == ticker]
    print(dff.head())
    
    dtemp1=dff[['p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y','p_chg6m','p_chg3m'
                , 's_chg10y','s_chg7y','s_chg5y','s_chg3y','s_chg2y','s_chg1y','s_chg6m','s_chg3m'
              ,'e_chg10y','e_chg7y','e_chg5y','e_chg3y','e_chg2y','e_chg1y','e_chg6m','e_chg3m']]

    dtemp2=dff[['ticker']]
    dtemp=dtemp2.join(dtemp1)
    dtemp=dtemp.set_index('ticker')
    
    df_s1=dtemp[['s_chg10y','s_chg7y','s_chg5y','s_chg3y','s_chg2y','s_chg1y','s_chg6m','s_chg3m']]
    df_s1=df_s1.rename({'s_chg10y': 'chg10y', 's_chg7y': 'chg7y','s_chg5y': 'chg5y',
                        's_chg3y': 'chg3y', 's_chg2y': 'chg2y','s_chg1y': 'chg1y',
                        's_chg6m': 'chg6m', 's_chg3m': 'chg3m'}
                        , axis=1)
    
    
    df_s2=dtemp[['e_chg10y','e_chg7y','e_chg5y','e_chg3y','e_chg2y','e_chg1y','e_chg6m','e_chg3m']]
    df_s2=df_s2.rename({'e_chg10y': 'chg10y', 'e_chg7y': 'chg7y','e_chg5y': 'chg5y',
                        'e_chg3y': 'chg3y', 'e_chg2y': 'chg2y','e_chg1y': 'chg1y',
                        'e_chg6m': 'chg6m', 'e_chg3m': 'chg3m'}
                        , axis=1)
     
    df_s3=dtemp[['p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y','p_chg6m','p_chg3m']]
    df_s3=df_s3.rename({'p_chg10y': 'chg10y', 'p_chg7y': 'chg7y','p_chg5y': 'chg5y',
                        'p_chg3y': 'chg3y', 'p_chg2y': 'chg2y','p_chg1y': 'chg1y',
                        'p_chg6m': 'chg6m', 'p_chg3m': 'chg3m'}
                        , axis=1)
   
    var=['s','e','p']
    
    df1=df_s1
    df1['stk']=df_s1.index
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['stk','indicator'],var_name='period',value_name='Rate')
    
    df2=df_s2
    df2['stk']=df_s2.index
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['stk','indicator'],var_name='period',value_name='Rate')
    
    df3=df_s3
    df3['stk']=df_s3.index
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['stk','indicator'],var_name='period',value_name='Rate')

    df = df1.append(df2)
    df = df.append(df3)
    
    print(df.head())
    
    fig = make_subplots(rows=1, cols=1,specs=[[{"secondary_y": True}]])


    colors = ['red','blue','purple','darkcyan','darkred']

    for i in range(len(var)): 
        sig=var[i]
        
        df_temp=df[df['indicator'].isin([sig])]
        
        if i<2:     
            fig.add_trace(
                go.Scatter(x=df_temp.period,
                           y=df_temp.Rate,
                           name=sig,
                            visible=True,
                            line=dict(width=1,color=colors[i]),
                        showlegend=True),row=1, col=1)
        else:
            fig.add_trace(
                go.Scatter(x=df_temp.period,
                           y=df_temp.Rate,
                           name=sig,
                            visible=True,
                            line=dict(width=1,color=colors[i]),
                        showlegend=True),secondary_y=True,row=1, col=1)
    fig.update_layout(title = ticker+' absolute s e and p (RHS) trend',
                  xaxis_title = 'period',
                  yaxis_title = 'trend'
                  )             
    
    return fig
       



@app.callback(Output('ts_plot4', 'figure'),
              [Input("scatter_plot2", "clickData"),Input('Scatter_data', 'data')])    



def update_timeseries4(clickData,jsonified_cleaned_data):
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    
    # print(clickData['points'][0]['text'])
    print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    dff = data[data['ticker'] == ticker]
    print(dff.head())
    
    
    dtemp1_rk=data[['p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y','p_chg6m','p_chg3m'
                , 's_chg10y','s_chg7y','s_chg5y','s_chg3y','s_chg2y','s_chg1y','s_chg6m','s_chg3m'
              ,'e_chg10y','e_chg7y','e_chg5y','e_chg3y','e_chg2y','e_chg1y','e_chg6m','e_chg3m']].rank(pct=True)
    dtemp2=data[['ticker']]
    dtemp_rk=dtemp2.join(dtemp1_rk)
    dtemp_rk = dtemp_rk[dtemp_rk['ticker'] == ticker]
    dtemp_rk=dtemp_rk.set_index('ticker')

    
    df_s1=dtemp_rk[['s_chg10y','s_chg7y','s_chg5y','s_chg3y','s_chg2y','s_chg1y','s_chg6m','s_chg3m']]
    df_s1=df_s1.rename({'s_chg10y': 'chg10y', 's_chg7y': 'chg7y','s_chg5y': 'chg5y',
                        's_chg3y': 'chg3y', 's_chg2y': 'chg2y','s_chg1y': 'chg1y',
                        's_chg6m': 'chg6m', 's_chg3m': 'chg3m'}
                        , axis=1)

    df_s2=dtemp_rk[['e_chg10y','e_chg7y','e_chg5y','e_chg3y','e_chg2y','e_chg1y','e_chg6m','e_chg3m']]
    df_s2=df_s2.rename({'e_chg10y': 'chg10y', 'e_chg7y': 'chg7y','e_chg5y': 'chg5y',
                        'e_chg3y': 'chg3y', 'e_chg2y': 'chg2y','e_chg1y': 'chg1y',
                        'e_chg6m': 'chg6m', 'e_chg3m': 'chg3m'}
                        , axis=1)
    
    df_s3=dtemp_rk[['p_chg10y','p_chg7y','p_chg5y','p_chg3y','p_chg2y','p_chg1y','p_chg6m','p_chg3m']]
    df_s3=df_s3.rename({'p_chg10y': 'chg10y', 'p_chg7y': 'chg7y','p_chg5y': 'chg5y',
                        'p_chg3y': 'chg3y', 'p_chg2y': 'chg2y','p_chg1y': 'chg1y',
                        'p_chg6m': 'chg6m', 'p_chg3m': 'chg3m'}
                        , axis=1)
    
    var=['s_rk','e_rk','p_rk']   
    
    df1=df_s1
    df1['stk']=df_s1.index
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['stk','indicator'],var_name='period',value_name='Rate')
    
    df2=df_s2
    df2['stk']=df_s2.index
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['stk','indicator'],var_name='period',value_name='Rate')
    
    df3=df_s3
    df3['stk']=df_s3.index
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['stk','indicator'],var_name='period',value_name='Rate')
    
    
    df_rk = df1.append(df2)
    df_rk = df_rk.append(df3)
        
    print(df_rk.head())
    
    fig = make_subplots(rows=1, cols=1)


    colors = ['red','blue','purple','darkcyan','darkred']

    for i in range(len(var)): 
        sig=var[i]
        
        df_temp=df_rk[df_rk['indicator'].isin([sig])]
             
        fig.add_trace(
            go.Scatter(x=df_temp.period,
                       y=df_temp.Rate,
                       name=sig,
                        visible=True,
                        line=dict(width=1,color=colors[i]),
                    showlegend=True),row=1, col=1)

    fig.update_layout(title = ticker+' relative s e and p trend',
                  xaxis_title = 'period',
                  yaxis_title = 'trend'
                  )    
    return fig



@app.callback(Output('bar_chart1', 'figure'),
              [Input("scatter_plot2", "clickData"),Input('Scatter_data', 'data')])    


def update_barchart1(clickData,jsonified_cleaned_data):
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    
    # print(clickData['points'][0]['text'])
    print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    dff = data[data['ticker'] == ticker]
    print(dff.head())
    
    dtemp=dff[['p_10y_cagr','p_7y_cagr','p_5y_cagr','p_3y_cagr','p_1y_cagr'
                ,'s_10y_cagr','s_7y_cagr','s_5y_cagr','s_3y_cagr','s_1y_cagr'
                ,'e_10y_cagr','e_7y_cagr','e_5y_cagr','e_3y_cagr','e_1y_cagr'
                ,'ebitda_10y_cagr','ebitda_7y_cagr','ebitda_5y_cagr','ebitda_3y_cagr','ebitda_1y_cagr'
                ,'PTX_10y_cagr','PTX_7y_cagr','PTX_5y_cagr','PTX_3y_cagr','PTX_1y_cagr']]
    
    
   
    df_s1=dtemp[['p_10y_cagr','p_7y_cagr','p_5y_cagr','p_3y_cagr','p_1y_cagr']]
    df_s1=df_s1.rename({'p_10y_cagr': '10y', 'p_7y_cagr': '7y','p_5y_cagr': '5y',
                        'p_3y_cagr': '3y', 'p_1y_cagr': '1y'}
                        , axis=1)

    df_s2=dtemp[['s_10y_cagr','s_7y_cagr','s_5y_cagr','s_3y_cagr','s_1y_cagr']]
    df_s2=df_s2.rename({'s_10y_cagr': '10y', 's_7y_cagr': '7y','s_5y_cagr': '5y',
                        's_3y_cagr': '3y', 's_1y_cagr': '1y'}
                        , axis=1)
    
    df_s3=dtemp[['e_10y_cagr','e_7y_cagr','e_5y_cagr','e_3y_cagr','e_1y_cagr']]
    df_s3=df_s3.rename({'e_10y_cagr': '10y', 'e_7y_cagr': '7y','e_5y_cagr': '5y',
                        'e_3y_cagr': '3y', 'e_1y_cagr': '1y'}
                        , axis=1)
    
    df_s4=dtemp[['ebitda_10y_cagr','ebitda_7y_cagr','ebitda_5y_cagr','ebitda_3y_cagr','ebitda_1y_cagr']]
    df_s4=df_s4.rename({'ebitda_10y_cagr': '10y', 'ebitda_7y_cagr': '7y','ebitda_5y_cagr': '5y',
                        'ebitda_3y_cagr': '3y', 'ebitda_1y_cagr': '1y'}
                        , axis=1)
    
    df_s5=dtemp[['PTX_10y_cagr','PTX_7y_cagr','PTX_5y_cagr','PTX_3y_cagr','PTX_1y_cagr']]
    df_s5=df_s5.rename({'PTX_10y_cagr': '10y', 'PTX_7y_cagr': '7y','PTX_5y_cagr': '5y',
                        'PTX_3y_cagr': '3y', 'PTX_1y_cagr': '1y'}
                        , axis=1)
    
    var=['p','s','e','ebitda','ptx']   
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df4=df_s4
    df4['indicator']=var[3]
    df4 = pd.melt(df_s4,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df5=df_s5
    df5['indicator']=var[4]
    df5 = pd.melt(df_s5,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df = df1.append(df2)
    df = df.append(df3)
    df = df.append(df4)
    df = df.append(df5)
  
        
    print(df.head())
    
    fig = px.bar(df, x="period", y="Rate", 
                 color="indicator", barmode='group')
    
    return fig


@app.callback(Output('bar_chart2', 'figure'),
              [Input("scatter_plot2", "clickData"),Input('Scatter_data', 'data')])    


def update_barchart2(clickData,jsonified_cleaned_data):
    data = pd.read_json(jsonified_cleaned_data, orient='split')
    # print(clickData['points'][0]['text'])
    print(clickData['points'][0])
    ticker = clickData['points'][0]['text']
    
    dtemp=data[['p_10y_cagr','p_7y_cagr','p_5y_cagr','p_3y_cagr','p_1y_cagr'
                ,'s_10y_cagr','s_7y_cagr','s_5y_cagr','s_3y_cagr','s_1y_cagr'
                ,'e_10y_cagr','e_7y_cagr','e_5y_cagr','e_3y_cagr','e_1y_cagr'
                ,'ebitda_10y_cagr','ebitda_7y_cagr','ebitda_5y_cagr','ebitda_3y_cagr','ebitda_1y_cagr'
                ,'PTX_10y_cagr','PTX_7y_cagr','PTX_5y_cagr','PTX_3y_cagr','PTX_1y_cagr']].rank(pct=True)
    dtemp2=data[['ticker']]
    dtemp=dtemp2.join(dtemp)
    
    dtemp = dtemp[dtemp['ticker'] == ticker]
    print(dtemp.head())
   
    df_s1=dtemp[['p_10y_cagr','p_7y_cagr','p_5y_cagr','p_3y_cagr','p_1y_cagr']]
    df_s1=df_s1.rename({'p_10y_cagr': '10y', 'p_7y_cagr': '7y','p_5y_cagr': '5y',
                        'p_3y_cagr': '3y', 'p_1y_cagr': '1y'}
                        , axis=1)

    df_s2=dtemp[['s_10y_cagr','s_7y_cagr','s_5y_cagr','s_3y_cagr','s_1y_cagr']]
    df_s2=df_s2.rename({'s_10y_cagr': '10y', 's_7y_cagr': '7y','s_5y_cagr': '5y',
                        's_3y_cagr': '3y', 's_1y_cagr': '1y'}
                        , axis=1)
    
    df_s3=dtemp[['e_10y_cagr','e_7y_cagr','e_5y_cagr','e_3y_cagr','e_1y_cagr']]
    df_s3=df_s3.rename({'e_10y_cagr': '10y', 'e_7y_cagr': '7y','e_5y_cagr': '5y',
                        'e_3y_cagr': '3y', 'e_1y_cagr': '1y'}
                        , axis=1)
    
    df_s4=dtemp[['ebitda_10y_cagr','ebitda_7y_cagr','ebitda_5y_cagr','ebitda_3y_cagr','ebitda_1y_cagr']]
    df_s4=df_s4.rename({'ebitda_10y_cagr': '10y', 'ebitda_7y_cagr': '7y','ebitda_5y_cagr': '5y',
                        'ebitda_3y_cagr': '3y', 'ebitda_1y_cagr': '1y'}
                        , axis=1)
    
    df_s5=dtemp[['PTX_10y_cagr','PTX_7y_cagr','PTX_5y_cagr','PTX_3y_cagr','PTX_1y_cagr']]
    df_s5=df_s5.rename({'PTX_10y_cagr': '10y', 'PTX_7y_cagr': '7y','PTX_5y_cagr': '5y',
                        'PTX_3y_cagr': '3y', 'PTX_1y_cagr': '1y'}
                        , axis=1)
    
    var=['p','s','e','ebitda','ptx']   
    
    df1=df_s1
    df1['indicator']=var[0]
    df1 = pd.melt(df_s1,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df2=df_s2
    df2['indicator']=var[1]
    df2 = pd.melt(df_s2,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df3=df_s3
    df3['indicator']=var[2]
    df3 = pd.melt(df_s3,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df4=df_s4
    df4['indicator']=var[3]
    df4 = pd.melt(df_s4,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df5=df_s5
    df5['indicator']=var[4]
    df5 = pd.melt(df_s5,id_vars=['indicator'],var_name='period',value_name='Rate')
    
    df = df1.append(df2)
    df = df.append(df3)
    df = df.append(df4)
    df = df.append(df5)
  
        
    print(df.head())
    
    fig = px.bar(df, x="period", y="Rate", 
                 color="indicator", barmode='group')
    
    return fig

      
if __name__ == '__main__':
    app.run_server(debug=False)
    
