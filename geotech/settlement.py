import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
from geotech import Layout as Id
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go

fig = px.line(x=[1, 2, 3, 4, 5], y=["a", "b", "c", "d", "e"], title="Test Graph")


tab = [dcc.Graph(id="settlement-graph-viz")]
