import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dtt

from datetime import datetime as dt
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd


start_date = dt(2017, 1, 1)


def fetch_data(stocks):
    end = dt.today()
    prices = yf.download(stocks, start=start_date, end=end)
    return prices


# stocks = ["SSO", "QLD", 'VWO', 'MBG', 'TLT', 'AGG', 'IAU', 'IEF', 'TMF', 'TQQQ']
stocks = ["MTUM", "USMV", 'EEMV', 'IAU', 'TLT', 'IEF', 'MBG']
# stocks = ["TQQQ", "TMF"]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = fetch_data(stocks)

# universe = [{'label': 'SSO', 'value': 'SSO'},
#             {'label': 'QLD', 'value': 'QLD'},
#             {'label': 'VTI', 'value': 'VTI'},
#             {'label': 'TLT', 'value': 'TLT'},
#             {'label': 'MBG', 'value': 'MBG'},
#             {'label': 'AGG', 'value': 'AGG'}]
universe = [{'label': stock, 'value': stock} for stock in stocks]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='To the moon'),
    html.Label(children='Sobesice Capital'),
    html.Div(children=[
        html.Label('Today is: {}'.format(dt.today())),
    ]),
    html.Div(id='output-container-date-picker-single'),
    html.Label('Assets:'),
    dcc.Dropdown(
        id='universe',
        options=universe,
        value=[],
        multi=True,
        placeholder='Choose the universe....and life duh',
        loading_state={}),
    html.Div(id='my-div'),
    # Price Graph
    dcc.Graph(
        figure=go.Figure(
            data=[go.Scatter(x=df.index.to_series(),
                             y=df['Adj Close'][stock], mode='lines', name=stock) for stock in stocks],
            layout=go.Layout(
                title='Universe Historical Data',
            )
        ),
        id='graph-prices'
    ),
    # Momentum Graph
    html.Div(children=[
        html.H3(children='Set Momentum'),
        html.Label('Momentum Lookback:'),
        dcc.Input(id='mom-lookback', value=110, type='number'),
        dcc.Graph(figure=go.Figure(
            data=[go.Scatter(x=df.index.to_series(),
                             y=df['Adj Close'][stock].rolling(100).apply(lambda x: (x[-1]) / x[0]), mode='lines',
                             name=stock) for stock in stocks],
            layout=go.Layout(
                title='Momentum History',
            )
        ),
            id='graph-mom')
    ]),

    # Vol Graph
    html.Div(children=[
        html.H3(children='Set Volatility'),
        html.Label('Volatility Lookback:'),
        dcc.Input(id='vol-lookback', value=21, type='number'),
        dcc.Graph(figure=go.Figure(
            data=[go.Scatter(x=df.index.to_series(),
                             y=df['Adj Close'][stock].rolling(100).apply(lambda x: x.std()), mode='lines',
                             name=stock) for stock in stocks],
            layout=go.Layout(
                title='Volatility History',
            )
        ),
            id='graph-vol')
    ]),

    # Calc Weights
    html.Div(children=[
        html.H3(children='Calculate Riskparity weights - this is where the T&W proprietary magic happens'),
        html.Label('Capital:'),
        dcc.Input(id='capital', value=10000, type='number'),
        html.Label('Trading day:'),
        dcc.DatePickerSingle(
            id='trading_day',
            min_date_allowed=start_date,
            max_date_allowed=dt.today(),
            initial_visible_month=dt.today(),
            date=str(dt.today())
        ),
        html.Label('Number of assets held:'),
        dcc.Input(id='k', value=4, type='number'),
        dtt.DataTable(id='table-results', columns = [{"name": i, "id": i} for i in stocks], data=[{}])
    ])])


# Calculate Weights React
@app.callback(
    [Output('table-results', 'data'), Output('table-results', 'columns')],
    [Input('k', 'value'), Input('mom-lookback', 'value'), Input('vol-lookback', 'value'), Input('trading_day', 'date'), Input('capital','value'), Input('universe', 'value')])
def update_columns(k, window_mom, window_vol, trading_day, capital, universe):
    df_filtred = df[df.index < trading_day]['Adj Close'][universe]
    moms = df_filtred.rolling(window_mom).apply(lambda x: (x[-1]) / x[0])
    assets_risk_budget = [1 / k] * k
    x0 = 1.0 * np.ones(k) / k
    bestmom = (moms.iloc[-1]).nlargest(k).index.values
    cov_mat = df_filtred[bestmom].pct_change().iloc[-window_vol:-1].cov()
    w = design_pf(cov_mat.values, assets_risk_budget, x0)
    cols = [{"name": i, "id": i} for i in list(bestmom)]
    alloc = np.multiply(capital,w)
    rdf = pd.DataFrame(np.array([w,alloc]),columns=bestmom)
    rows = rdf.to_dict('rows')
    print(rows)
    return rows, cols


# Graph volatility REACT
@app.callback(
    Output('graph-vol', 'figure'),
    [Input('vol-lookback', 'value'), Input('universe', 'value')])
def update_figure(lookback, assets):
    traces = [go.Scatter(x=df.index.to_series(),
                         y=df['Adj Close'][stock].pct_change().rolling(lookback).apply(lambda x: x.std()), mode='lines',
                         name=stock) for stock in assets]
    return {
        'data': traces,
        'layout': go.Layout(title='Returns Volatility History')
    }


# Graph momentum REACT
@app.callback(
    Output('graph-mom', 'figure'),
    [Input('mom-lookback', 'value'), Input('universe', 'value')])
def update_figure(lookback, assets):
    traces = [go.Scatter(x=df.index.to_series(),
                         y=df['Adj Close'][stock].rolling(lookback).apply(lambda x: (x[-1]) / x[0]), mode='lines',
                         name=stock) for stock in assets]
    return {
        'data': traces,
        'layout': go.Layout(title='Momentum History')
    }


# Graph prices REACT
@app.callback(
    Output('graph-prices', 'figure'),
    [Input('universe', 'value')])
def update_figure(assets):
    traces = [go.Scatter(x=df.index.to_series(),
                         y=df['Adj Close'][stock], mode='lines', name=stock) for stock in assets]
    return {
        'data': traces,
        'layout': go.Layout(title='Universe Historical Data')
    }


# Naive Callback
@app.callback(Output(component_id='my-div', component_property='children'),
              [Input(component_id='universe', component_property='value')])
def update_output_div(input_value):
    return 'Universe selected: "{}"'.format(input_value)




if __name__ == '__main__':
    print('Honiklady holky rady')
    app.run_server(debug=True   , host = '0.0.0.0', port=5000)
