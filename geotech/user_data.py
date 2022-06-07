import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table
from dash import dcc
from geotech import Layout as Id
from dash.dependencies import Input, Output, State
from geotech import app
import pandas as pd


def text_input(identity):
    return (
        dcc.Input(
            type="text",
            placeholder="Upload your file here",
            readOnly=True,
            id=identity,
        ),
    )


# Water and soil value configurations
water_input = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Label("Unit wait of water"),
                dbc.Input(type="text", id="water_weight", value=9.81),
            ]
        ),
        dbc.Col(
            [
                dbc.Label("Water level (mbgl)"),
                dbc.Input(type="text", id="water_level", value=0),
            ]
        ),
    ]
)

soil_input = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Label("Unit weight of soil"),
                dbc.Input(type="text", id="soil_weight", value=0),
            ]
        ),
        dbc.Col(
            [
                dbc.Label("Unit weight of fill"),
                dbc.Input(type="text", id="fill_weight", value=0),
            ]
        ),
    ]
)

misc_input = dbc.Row(
    [
        dbc.Col(
            [
                dbc.Label("Ratio of su to s"),
                dbc.Input(type="text", id="ratio_su_s", value=0),
            ]
        ),
        dbc.Col(
            [dbc.Label("VWP7 depth (mbgl)"), dbc.Input(type="text", id="vwp7", value=0)]
        ),
    ]
)


annotation_input = dbc.Row(
    [
        dbc.Label("Annotation", width="3", className="mmCard-label"),
        dbc.Col(
            dcc.Input(
                type="text",
                placeholder="Upload your file here",
                id="annotation-file-name",
            ),
            width="4",
        ),
        dbc.Col(dcc.Upload(id=Id.upload.annotation, children=(html.Button("Upload")))),
    ],
    className="mb-3",
)
settlement_input = dbc.Row(
    [
        dbc.Label("Settlement", width="3", className="mmCard-label"),
        dbc.Col(
            text_input("settlement-file-name"),
            width="4",
        ),
        dbc.Col(
            dcc.Upload(
                id=Id.upload.settlement,
                children=(html.Button("Upload", className="mmUpload")),
            )
        ),
    ],
    className="mb-3",
)
construction_input = dbc.Row(
    [
        dbc.Label("Construction", width="3", className="mmCard-label"),
        dbc.Col(
            text_input("construction-file-name"),
            width="4",
        ),
        dbc.Col(
            dcc.Upload(
                id=Id.upload.construction,
                children=(html.Button("Upload", className="mmUpload")),
            )
        ),
    ],
    className="mb-3",
)
piezo_input = dbc.Row(
    [
        dbc.Label("Piezo", width="3", className="mmCard-label"),
        dbc.Col(
            text_input("piezo-file-name"),
            width="4",
        ),
        dbc.Col(
            dcc.Upload(
                id=Id.upload.piezo,
                children=(html.Button("Upload", className="mmUpload")),
            )
        ),
    ],
    className="mb-3",
)
crest_input = dbc.Row(
    [
        dbc.Label("Crest", width="3", className="mmCard-label"),
        dbc.Col(
            text_input("crest-file-name"),
            width="4",
        ),
        dbc.Col(
            dcc.Upload(
                id=Id.upload.crest,
                children=(html.Button("Upload", className="mmUpload")),
            )
        ),
    ],
    className="mb-3",
)
toe_input = dbc.Row(
    [
        dbc.Label("Toe", width="3", className="mmCard-label"),
        dbc.Col(
            text_input("toe-file-name"),
            width="4",
        ),
        dbc.Col(
            dcc.Upload(
                id=Id.upload.toe, children=(html.Button("Upload", className="mmUpload"))
            )
        ),
    ],
    className="mb-3",
)

# file_selections = dbc.Card(
#     dbc.CardBody(
#         [
#             html.Br(),
#             dbc.CardHeader("Choose your data format", className="card-title"),
#             dbc.Checklist(
#                 options=[
#                     {"label": "csv", "value": ".csv"},
#                     {"label": "xlsx", "value": ".xlsx"},
#                 ],
#                 value=[],
#                 id=Id.fileselection,
#                 inline=True,
#                 switch=True,
#             ),
#         ]
#     ),
#     color="light",
#     outline=True,
# )

file_uploads = dbc.Card(
    dbc.CardBody(
        [
            dbc.CardHeader("Upload your data below ", className="mm-card-title"),
            html.Br(),
            annotation_input,
            settlement_input,
            construction_input,
            piezo_input,
            crest_input,
            toe_input,
        ]
    ),
    color="light",
    outline=True,
)

misc_options = dbc.Card(
    dbc.CardBody(
        [
            dbc.CardHeader(
                "Enter your paramer values here.", className="mm-card-title"
            ),
            html.Br(),
            water_input,
            soil_input,
            misc_input,
        ]
    ),
)
cols = [
    {"id": "param", "name": "Parameter"},
    {"id": "settlement", "name": "Settlement", "presentation": "dropdown"},
    {"id": "construction", "name": "Construction", "presentation": "dropdown"},
    {"id": "piezo", "name": "Piezo", "presentation": "dropdown"},
    {"id": "crest", "name": "Crest", "presentation": "dropdown"},
    {"id": "toe", "name": "Toe", "presentation": "dropdown"},
]

user_config = dbc.Card(
    dbc.CardBody(
        dash_table.DataTable(
            id=Id.Tabs.UserData.config,
            columns=cols,
            editable=True,
            data=[{"param": "ID"}, {"param": "Date Time"}, {"param": "Data"}],
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
            style_cell_conditional=[
                {"if": {"column_id": "date-time"}, "textAlign": "center"},
                {"if": {"column_id": "data"}, "textAlign": "left"},
            ],
            # alternate rows are coloured
            style_data_conditional=[
                {"if": {"row_index": "even"}, "backgroundColor": "#EDEBE6"}
            ],
        )
    ),
    style={"width": "90%"},
)

tab = [
    # dbc.Row(
    #     [
    #         dbc.Col(file_selections, className="mmCard", width="2"),
    #     ],
    #     id="main-user-data1",
    # ),
    dbc.Row(
        [
            dbc.Col(misc_options, className="mmCard", width="5"),
            dbc.Col(file_uploads, className="mmCard", width="5"),
        ],
        id="misc_user_data2",
    ),
    dbc.Row(
        [
            dbc.Col(user_config, className="mmCard"),
        ],
        id="misc_user_data3",
    ),
]
