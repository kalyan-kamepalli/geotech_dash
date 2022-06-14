from dash import html
import dash_bootstrap_components as dbc
from dash import dcc
from geotech import Layout as ID
from geotech import (
    home,
    user_data,
    settlement,
    piezo,
    crest_toe,
    guo_chu,
    porewater,
    asaoka,
)

navbar = dbc.Navbar(
    [
        dbc.Col(
            [
                dbc.NavbarBrand(
                    html.Img(src="./assets/mm_logo.png", height="80px"),
                )
                # html.Div("Geo Tech", className="mmTitle"),
            ],
            width=1,
            style={"background": "#1ad1cb"},
        ),
        dbc.Col(
            [
                dbc.Row("Instrumentation & Monitoring", className="mmTitle"),
                dbc.Row("Data Interpretation Tool", className="mmSubtitle"),
            ],
            width=3,
            style={"background": "#1ad1cb"},
        ),
        dbc.Col(
            dbc.Row(
                #         [
                #             dbc.Col(
                #                 dbc.Button("Home", id=ID.NavBar.BtnHome, className="mmButton"),
                #                 width="auto",
                #             ),
                #             dbc.Col(
                #                 dbc.Button(
                #                     "Data Uploads",
                #                     id=ID.NavBar.BtnData,
                #                     className="mmButton",
                #                 ),
                #                 width="auto",
                #             ),
                #             dbc.Col(
                #                 dbc.Button(
                #                     "Visualizations",
                #                     id=ID.NavBar.BtnViz,
                #                     className="mmButton",
                #                 ),
                #                 width="auto",
                #             ),
                #         ],
                align="center",
                justify="end",
                style={"height": "80px"},
            ),
            width=8,
            style={"background": "#333333", "height": "80px", "align-content": "end"},
        ),
    ],
    #  no_gutters=True,
    #  justify='start'
    # ),
    color="#FFFFFF",
    dark=True,
)

tabs = [
    dbc.Tab(
        home.tab,
        tab_id=ID.Tabs.Home.Id,
        label="Home",
        active_label_style={"color": "#FFFFFF", "background": "#b270c8"},
        label_class_name="tablabel",
    ),
    dbc.Tab(
        user_data.tab,
        tab_id=ID.Tabs.UserData.Id,
        label="Configurations",
        active_label_style={"color": "#FFFFFF", "background": "#b270c8"},
        label_class_name="tablabel",
    ),
    dbc.Tab(
        settlement.tab,
        tab_id=ID.Tabs.Visualizations.settlement,
        label="Settlement",
        active_label_style={"color": "#FFFFFF", "background": "#b270c8"},
        label_class_name="tablabel",
    ),
    dbc.Tab(
        piezo.tab,
        tab_id=ID.Tabs.Visualizations.piezo,
        label="Piezo",
        active_label_style={"color": "#FFFFFF", "background": "#b270c8"},
        label_class_name="tablabel",
    ),
    dbc.Tab(
        crest_toe.tab,
        tab_id=ID.Tabs.Visualizations.crest,
        label="Crest & Toe",
        active_label_style={"color": "#FFFFFF", "background": "#b270c8"},
        label_class_name="tablabel",
    ),
    dbc.Tab(
        porewater.tab,
        tab_id=ID.Tabs.Visualizations.porewater,
        active_label_style={"color": "#FFFFFF", "background": "#b270c8"},
        label="Porewater Pressure",
        label_class_name="tablabel",
    ),
    dbc.Tab(
        asaoka.tab,
        tab_id=ID.Tabs.Visualizations.asaoka,
        label="Asaoka",
        active_label_style={"color": "#FFFFFF", "background": "#b270c8"},
        label_class_name="tablabel",
    ),
    dbc.Tab(
        guo_chu.tab,
        tab_id=ID.Tabs.Visualizations.guo,
        label="Guo & Chu",
        active_label_style={"color": "#FFFFFF", "background": "#b270c8"},
        label_class_name="tablabel",
    ),
]

input_box = html.Div(
    dbc.Tabs(tabs, class_name="tabs-style", id=ID.Tabs.Id), className="ui-box"
)


def layout():

    content = [
        dcc.Location(id=ID.Reload, refresh=False),
        dcc.Store(id=ID.upload.AnnotationsData, storage_type="session"),
        dcc.Store(id=ID.upload.SettlementData, storage_type="session"),
        dcc.Store(id=ID.upload.ConstructionData, storage_type="session"),
        dcc.Store(id=ID.upload.PiezoData, storage_type="session"),
        dcc.Store(id=ID.upload.CrestData, storage_type="session"),
        dcc.Store(id=ID.upload.ToeData, storage_type="session"),
        navbar,
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(input_box, id=ID.ColumnData),
                    ],
                    id=ID.Main.MainRow,
                    className="h-100 flex-fill d-flex",
                    style={"background": "#FFFFFF"},
                ),
            ],
            fluid=True,
            className="d-flex h-100 flex-column",
        ),
    ]
    return html.Div(
        content, className="d-flex h-100 flex-column", style=dict(background="#FFFFFF")
    )
