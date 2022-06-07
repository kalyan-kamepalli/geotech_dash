import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table

from geotech import Layout as Id

tab = [
    html.Div(id="main-container"),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H3(
        "This data interpretation tool was prepared as part of the Digital Committee PMO’s Optimisation Programme by Fabien Blanchais and Kalyan Kamepalli. The tool is a practice approved tool of the Geotechnics Practice.",
        className="information",
    ),
    html.Br(),
    html.H3("The tool is producing the following plots:", className="information"),
    html.Br(),
    html.H3(
        "Settlement vs. Time in relation to the height of the embankment",
        className="info",
    ),
    html.H3(
        "Piezo level vs. Time in relation to the height of the embankment",
        className="info",
    ),
    html.H3(
        "Crest and toe movement vs. Time in relation to the height of the embankment",
        className="info",
    ),
    html.H3(
        "Excess Pore Pressure vs. Time in relation to the height of the embankment",
        className="info",
    ),
    html.H3(
        "Factor of Safety vs. Time in relation to the height of the embankment",
        className="info",
    ),
    html.H3(
        "Ultimate settlement based on Asaoka (1978)",
        className="info",
    ),
    html.H3(
        "Ultimate settlement based on Guo and Chu (1977)",
        className="info",
    ),
    html.Br(),
    html.Br(),
    html.H3("References used:", className="minorInfo"),
    html.H3(
        "Asaoka, A. (1978). Observational Procedure of Settlement Prediction. Soils and Foundations, 18(4),pp87-101",
        className="minorInfo",
    ),
    html.H3(
        "Matuso, M & Kawamura, K.(1977). Diagram for construction control of embankment on soft ground. Solid and Foundations, 17(3), pp37-52",
        className="minorInfo",
    ),
]
