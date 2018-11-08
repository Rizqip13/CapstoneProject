import dash
import dash_core_components as dcc 
import dash_html_components as html  
import pandas as pd  
import plotly.graph_objs as go  
from dash.dependencies import Input, Output
import numpy as np
import pickle
import category_encoders as ce

pd.set_option('display.max_columns',500)
df= pd.read_csv('train.csv')
dfDescribe= pd.read_csv('describe.csv')
XGBmodel = pickle.load(open('merc_benzXGB.sav','rb'))
coef = pd.read_csv('feature.csv')
data=df.drop('y',axis=1)


listcol=[]
for x in df.iloc[:,2:].columns:
    listcol.append({'label':x,'value':x})

listID=[]
for x in df['ID']:
    listID.append({'label':str(x),'value':str(x)})

catopt=[]
for x in dfDescribe[dfDescribe['Variable type']=='Categorical']['dataFeatures'].values:
    catopt.append({'label':x,'value':x})

# for running the model

encoder = ce.BinaryEncoder(cols=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'])

def runmodel(test, ID):
    test=test.drop(dfDescribe[dfDescribe['unique']==1]['dataFeatures'].values,axis=1)
    Xtest = encoder.fit_transform(test)
    temp = Xtest[Xtest['ID']==int(ID)]
    return XGBmodel.predict(temp)

app = dash.Dash(__name__)
server = app.server


def ddltab1():
    return html.Table(className='ddl-table', children=[
                html.Tr([
                    html.Td([
                        dcc.Dropdown(
                            id='ddl-table',
                            options=[{'label':'The Data','value':'Dataset'},
                                    {'label':'Dataset Columns Describe','value':'Columns-Describe'}],
                            value='Dataset'
                        )
                    ])
                ])
            ])

def ddltab2():
    return html.Table(className='ddl-cat', children=[
                html.Tr([
                    html.Td([
                        dcc.Dropdown(
                            id='ddl-cat',
                            options=catopt,
                            value='X0'
                        )
                    ])
                ])
            ])

def ddltab4():
    return html.Table(className='ddl-mod'[
        html.Tr([
            html.Td([
                dcc.Dropdown(
                id='ddl-ID',
                options=listID,
                value='0'
                )
            ])
        ]),
        html.Tr([
            html.Td([
                dcc.Dropdown(
                    id='ddl-col',
                    options=listcol,
                    value='X0'
                )
            ]),
            html.Td([
                html.H1('test')
            ])
        ])
    ])

app.tittle='Mercedes-Benz Greener Manufacturing'

app.layout = html.Div(className='utama',children=[
    dcc.Tabs(id="Tabs",value='tab-1',children=[
        dcc.Tab(label='Mercedes-Benz Dataset',value='tab-1', children=[
            #isi Tab 1
            ddltab1(),
            html.H4('Columns Slider'),
            dcc.Slider(
                id='columns-slider',
                min=2,
                max=len(df.columns)-12,
                value=3
            ),
            html.Div(id='dataset-container')
        ]),
        dcc.Tab(label='Categorical Plot',value='tab-2',children=[
            #isi Tab 2
            ddltab2(),
            dcc.Graph(
                id='categoricalPlot',
                figure={}
            )

        ]),
        dcc.Tab(label='XGBoost',value='tab-3',children=[
            #isi Tab 3
            dcc.Graph(
                id='featureimportances',
                figure={'data':[go.Bar(
                        x=coef['feature'],
                        y=coef['coef'],
                        opacity=0.7
                    )],
                    'layout': go.Layout(
                        title='Feature Importances'
                    )
                    }
            )
        ]),
        dcc.Tab(label='Predictions',value='tab-4',children=[
            #isi Tab 4
            html.Div([
                html.H4('Id :'),
                dcc.Dropdown(
                id='ddl-ID',
                options=listID,
                value='0'
                )
            ],style={
                'width':'300px'
            }),
            html.H4('Feature Values that you want to changes'),
            html.Table([
                html.Tr([
                    html.Td([
                        html.H5('Feature :'),
                        dcc.Dropdown(
                            id='ddl-col',
                            options=listcol,
                            value='X0'
                        )
                    ],style={'width':'300px'}),
                    html.Td([
                        html.H5('Values : '),
                        dcc.Dropdown(
                            id='ddl-value'
                        )
                    ],style={'width':'300px'})
                ])
            ]),
            html.Div(id='hasilprediksi')
        ]),
    ],style={
        'fontFamily':'system-ui'
    },
    content_style={
        'fontFamily':'Arial',
        'borderLeft':'1px solid #d6d6d6',
        'borderRight':'1px solid #d6d6d6',
        'borderBottom':'1px solid #d6d6d6',
        'padding':'44px'
    })
])

@app.callback(
    Output('dataset-container','children'),
    [Input('ddl-table','value'),
    Input('columns-slider','value')]
)
def table(ddlinput,sliderinput):
    if (ddlinput=='Dataset'):
        data=pd.concat([df.iloc[:,:2],df.iloc[:,sliderinput:sliderinput+12]],axis=1) 
    else :
        data = dfDescribe
    return [
        html.H1(ddlinput, ' Table'),
        dcc.Graph(
            id='tableData',
            figure={
                'data':[go.Table(
                    header=dict(
                        values=['<b>' +col.capitalize() + '</b>' for col in data.columns],
                        fill = dict(color='#C2D4FF'),
                        font = dict(size=17),
                        height = 30,
                        align = ['center'] ),
                    cells=dict(
                        values=[data[col] for col in data.columns],
                        fill = dict(color='#F5F8FF'),
                        font = dict(size=15),
                        height = 25,
                        align = ['right'] * 5)
                )],
                'layout': dict(height=500,margin={'l': 40, 'b': 40, 't': 10, 'r': 10})
            }
        )
    ] 

@app.callback(
    Output('columns-slider','disabled'),
    [Input('ddl-table','value')]
)
def enable(val):
    if val == 'Dataset':
        return False
    else :
        return True

@app.callback(
    Output('categoricalPlot','figure'),
    [Input('ddl-cat','value')]
)
def plot(cat):
    return {'data':[go.Box(
            x=df[str(cat)],
            y=df['y'],
            opacity=0.7
        )],
            'layout': go.Layout(
            title='Box Plot of '+str(cat)+' Category',
            xaxis={'title': str(cat)},
            yaxis={'title': 'y (time)'},
            # margin={'l': 40, 'b': 40, 't': 10, 'r': 10}
        )}

@app.callback(
    Output('ddl-value','options'),
    [Input('ddl-col','value')]
)
def dropdown(col):
    listvalues=[]
    for x in df[col].unique():
        listvalues.append({'label':x,'value':x})
    return listvalues

@app.callback(
    Output('hasilprediksi','children'),
    [Input('ddl-ID','value'),
    Input('ddl-col','value'),
    Input('ddl-value','value')]
) 
def hasil(ID, col, X):
    temp=data.copy()

    temp.loc[data[data['ID']==int(ID)].index,col]=X
    # runmodel(temp, ID)
    return html.H2('y = '+str(runmodel(temp,ID)))



if __name__ == '__main__': 
    #run server on port 1996
    #debug=True for auto restart if code editedheader=dict(
    app.run_server(debug=True, port=1996)