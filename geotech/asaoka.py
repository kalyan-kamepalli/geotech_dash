import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table
from dash import dcc

from geotech import Layout as Id

cols = [
    {"id": "instrument", "name": "Instrument"},
    {"id": "gradient", "name": "Gradient"},
    {"id": "interceptor", "name": "Intercept"},
    {"id": "ultimate_settlement", "name": "Ultimate settlement"},
    {"id": "diff_to_current", "name": "Difference to current"},
]

tab = [
    dbc.Row([dbc.Col(dcc.Graph(id="asaoka-graph"))]),
    dbc.Row(
        [
            dbc.Col(
                dash_table.DataTable(
                    id=Id.Tabs.Asaoka.table,
                    columns=cols,
                    editable=True,
                    # page definitions. page_action=native gives numbered pages
                    page_action="native",
                    page_current=0,
                    page_size=5,
                    style_header={
                        "backgroundColor": "#575756",
                        "color": "white",
                        "fontWeight": "bold",
                        "textAlign": "center",
                    },
                    style_cell={"font-family": "proximanova", "fontSize": 14},
                    # alternate rows are coloured
                    style_data_conditional=[
                        {"if": {"row_index": "even"}, "backgroundColor": "#EDEBE6"}
                    ],
                )
            )
        ]
    ),
]
