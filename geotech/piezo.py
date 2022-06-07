import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
from geotech import Layout as Id
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go

from geotech import Layout as Id

tab = [dcc.Graph(id="piezo-graph")]
