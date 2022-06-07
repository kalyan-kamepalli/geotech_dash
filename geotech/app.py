# builtin
import os, math, io, sys
from datetime import datetime, timedelta
from dash.dependencies import Input, Output, State
from geotech import Layout as Id
from base64 import b64encode, b64decode

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd

# 3rd party
import dash
from dash import ctx
import dash_bootstrap_components as dbc
from geotech import layout_main as lm

from plotly.subplots import make_subplots
import plotly.graph_objects as go

external_scripts = [
    {"src": "https://kit.fontawesome.com/b3853d35d7.js", "crossorigin": "anonymous"}
]

app = dash.Dash(
    __name__,
    title="GeoTeck Interpretation Tool",
    external_stylesheets=[dbc.themes.COSMO],
    external_scripts=external_scripts,
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
)

app.layout = lm.layout


@app.callback(
    Output(Id.upload.AnnotationsData, "data"),
    Input(Id.upload.annotation, "contents"),
    Input(Id.upload.annotation, "filename"),
)
def annotations_data(contents, file_name):
    content_type, content_string = contents.split(",")
    decoded = b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode("utf-8"))).sort_index()
    data = {"data": df.to_dict("records")}

    return data


def upload_data(data_stream):

    content_type, content_string = data_stream.split(",")

    decoded = b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    config = [{"label": col, "value": col} for col in df.columns]

    return df, config


@app.callback(
    Output("settlement-file-name", "placeholder"),
    Input(Id.upload.settlement, "filename"),
    Input(Id.upload.settlement, "contents"),
)
def show_settlement_file_name(filename, contents):
    return filename


@app.callback(
    Output("construction-file-name", "placeholder"),
    Input(Id.upload.construction, "filename"),
    Input(Id.upload.construction, "contents"),
)
def show_construction_file_name(filename, contents):
    return filename


@app.callback(
    Output("piezo-file-name", "placeholder"),
    Input(Id.upload.piezo, "filename"),
    Input(Id.upload.piezo, "contents"),
)
def show_piezo_file_name(filename, contents):
    return filename


@app.callback(
    Output("crest-file-name", "placeholder"),
    Input(Id.upload.crest, "filename"),
    Input(Id.upload.crest, "contents"),
)
def show_crest_file_name(filename, contents):
    return filename


@app.callback(
    Output("toe-file-name", "placeholder"),
    Input(Id.upload.toe, "filename"),
    Input(Id.upload.toe, "contents"),
)
def show_toe_file_name(filename, contents):
    return filename


@app.callback(
    Output("annotation-file-name", "placeholder"),
    Input(Id.upload.annotation, "filename"),
    Input(Id.upload.annotation, "contents"),
)
def show_annotation_file_name(filename, contents):
    return filename


@app.callback(
    Output(Id.Tabs.UserData.config, "dropdown"),
    Output(Id.upload.SettlementData, "data"),
    Output(Id.upload.ConstructionData, "data"),
    Output(Id.upload.PiezoData, "data"),
    Output(Id.upload.CrestData, "data"),
    Output(Id.upload.ToeData, "data"),
    Input(Id.upload.settlement, "filename"),
    Input(Id.upload.settlement, "contents"),
    Input(Id.upload.construction, "filename"),
    Input(Id.upload.construction, "contents"),
    Input(Id.upload.piezo, "filename"),
    Input(Id.upload.piezo, "contents"),
    Input(Id.upload.crest, "filename"),
    Input(Id.upload.crest, "contents"),
    Input(Id.upload.toe, "filename"),
    Input(Id.upload.toe, "contents"),
    State(Id.upload.SettlementData, "data"),
    State(Id.upload.ConstructionData, "data"),
    State(Id.upload.PiezoData, "data"),
    State(Id.upload.CrestData, "data"),
    State(Id.upload.ToeData, "data"),
)
def generate_config_table(
    sf,
    sdata,
    cf,
    cdata,
    pf,
    pdata,
    cs,
    csdata,
    tf,
    tdata,
    settlement,
    construction,
    piezo,
    crest,
    toe,
):
    trigger = ctx.triggered_id

    if trigger == "btn-settlement":
        (df, config) = upload_data(sdata)

        return (
            {
                "settlement": {"options": config},
                "construction": {"options": get_config_from(construction)},
                "piezo": {"options": get_config_from(piezo)},
                "crest": {"options": get_config_from(crest)},
                "toe": {"options": get_config_from(toe)},
            },
            {"data": df.to_dict("records")},
            construction,
            piezo,
            crest,
            toe,
        )

    if trigger == "btn-construction":
        (df, config) = upload_data(cdata)

        return (
            {
                "settlement": {"options": get_config_from(settlement)},
                "construction": {"options": config},
                "piezo": {"options": get_config_from(piezo)},
                "crest": {"options": get_config_from(crest)},
                "toe": {"options": get_config_from(toe)},
            },
            settlement,
            {"data": df.to_dict("records")},
            piezo,
            crest,
            toe,
        )

    if trigger == "btn-piezo":
        (df, config) = upload_data(pdata)

        return (
            {
                "settlement": {"options": get_config_from(settlement)},
                "construction": {"options": get_config_from(construction)},
                "piezo": {"options": config},
                "crest": {"options": get_config_from(crest)},
                "toe": {"options": get_config_from(toe)},
            },
            settlement,
            construction,
            {"data": df.to_dict("records")},
            crest,
            toe,
        )

    if trigger == "btn-crest":
        (df, config) = upload_data(csdata)

        return (
            {
                "settlement": {"options": get_config_from(settlement)},
                "construction": {"options": get_config_from(construction)},
                "piezo": {"options": get_config_from(piezo)},
                "crest": {"options": config},
                "toe": {"options": get_config_from(toe)},
            },
            settlement,
            construction,
            piezo,
            {"data": df.to_dict("records")},
            toe,
        )

    if trigger == "btn-toe":
        (df, config) = upload_data(tdata)

        return (
            {
                "settlement": {"options": get_config_from(settlement)},
                "construction": {"options": get_config_from(construction)},
                "piezo": {"options": get_config_from(piezo)},
                "crest": {"options": get_config_from(crest)},
                "toe": {"options": config},
            },
            settlement,
            construction,
            piezo,
            crest,
            {"data": df.to_dict("records")},
        )


def get_config_from(data):
    if data:
        return [
            {"label": col, "value": col}
            for col in pd.DataFrame.from_dict(data["data"]).columns
        ]
    return []


@app.callback(
    Output("settlement-graph-viz", "figure"),
    Input(Id.Tabs.Id, "active_tab"),
    Input(Id.Tabs.UserData.config, "data"),
    State(Id.upload.SettlementData, "data"),
    State(Id.upload.ConstructionData, "data"),
    State(Id.upload.AnnotationsData, "data"),
)
def settlement_graph(tab_name, rows, settlement, load, annotations):

    if tab_name == Id.Tabs.Visualizations.settlement:
        settlement_df = pd.DataFrame.from_dict(settlement["data"])
        load_df = pd.DataFrame.from_dict(load["data"])
        comments_df = pd.DataFrame.from_dict(annotations["data"])

        # rows are always 3 - ID, data time and Data with 5 fixed columns
        # column names - settlement, construction, piezo, crest and toe
        # construct a config object from user selection
        config = get_config(rows, "settlement")

        return get_figure(settlement_df, load_df, comments_df, config)

    return dash.no_update


def get_figure(data, load, annotations, config):

    # Convert date data to date first format
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    load["Date"] = pd.to_datetime(load["Date"], dayfirst=True)

    # re-index dataframe
    data = data.set_index("Date").sort_index()
    load = load.set_index("Date").sort_index()

    min_max_dh = (math.inf, -math.inf, pd.Timestamp.max, pd.Timestamp.min)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[90, 50],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    for identity in data[config["id"]].unique():

        df = data[(data[config["id"]] == identity)]
        df = df.dropna()

        df["dh"] = df[config["data"]].diff() / df.index.to_series().diff().astype(
            "timedelta64[D]"
        ).astype("Int64")

        df = df.dropna()

        fig.add_trace(
            go.Scatter(x=df.index, y=df[config["data"]], name=identity),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df["dh"], name=identity + "-dh"), row=2, col=1
        )

        min_max_dh_loop = min_max(df, "dh")
        min_max_dh = min_max_in_tuples(min_max_dh, min_max_dh_loop)

    # Use construction data here
    # get the state of construction data

    fig.add_trace(
        go.Scatter(
            x=load.index,
            y=load["Height"],
            name=load.iloc[0, 0],
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    min_max_df = min_max(data, config["data"])
    min_max_load = min_max(load, "Height")
    min_max_tuples = min_max_in_tuples(min_max_df, min_max_load)

    y_range = min_max_df[1] - min_max_df[0]
    y_add = (10 * y_range) / 100

    x_range = (min_max_tuples[3] - min_max_tuples[2]).days
    x_add = (2 * x_range) / 100
    x_add = timedelta(days=x_add)

    try:
        for index, row in annotations.iterrows():

            xs = pd.to_datetime(row[0], infer_datetime_format=True)
            xe = pd.to_datetime(row[0], infer_datetime_format=True)

            fig.add_trace(
                go.Scatter(
                    x=[xs, xe],
                    y=[min_max_df[0], min_max_df[1]],
                    name=row[1],
                    opacity=0.5,
                    showlegend=False,
                    line=dict(color="grey", dash="dash"),
                ),
                row=1,
                col=1,
            )

            fig.add_annotation(
                text=row[1],
                x=xs - x_add,
                y=min_max_df[0] + y_add,
                showarrow=False,
                textangle=-90,
                opacity=0.5,
                font=dict(color="grey"),
            )

            fig.add_trace(
                go.Scatter(
                    x=[xs, xe],
                    y=[min_max_dh[0], min_max_dh[1]],
                    name=row[1],
                    opacity=0.5,
                    showlegend=False,
                    line=dict(color="grey", dash="dash"),
                ),
                row=2,
                col=1,
            )
    except Exception as error:
        pass

    fig.update_layout(
        height=700,
        xaxis2_title="Time",
        yaxis1_title="Settlement (mm)",
        yaxis2_title="Construction height (m)",
        yaxis3_title="dh (mm/day)",
    )

    return fig


@app.callback(
    Output("piezo-graph", "figure"),
    Input(Id.Tabs.Id, "active_tab"),
    Input(Id.Tabs.UserData.config, "data"),
    State(Id.upload.PiezoData, "data"),
    State(Id.upload.ConstructionData, "data"),
    State(Id.upload.AnnotationsData, "data"),
)
def piezo_graph(tab_name, config_data, piezo_data, load, annotations):

    if tab_name == Id.Tabs.Visualizations.piezo:
        piezo_df = pd.DataFrame.from_dict(piezo_data["data"])
        load_df = pd.DataFrame.from_dict(load["data"])
        comments_df = pd.DataFrame.from_dict(annotations["data"])

        config = get_config(config_data, "piezo")

        return get_figure(piezo_df, load_df, comments_df, config)

    return dash.no_update


@app.callback(
    Output("crest-toe-graph", "figure"),
    Input(Id.Tabs.Id, "active_tab"),
    State(Id.upload.CrestData, "data"),
    State(Id.upload.ToeData, "data"),
    State(Id.upload.ConstructionData, "data"),
    State(Id.upload.AnnotationsData, "data"),
)
def crest_toe_graph(tab_name, crest_data, toe_data, load, annotations):

    if tab_name == Id.Tabs.Visualizations.crest:

        crest_df = pd.DataFrame.from_dict(crest_data["data"])
        toe_df = pd.DataFrame.from_dict(toe_data["data"])
        load_df = pd.DataFrame.from_dict(load["data"])
        comments_df = pd.DataFrame.from_dict(annotations["data"])

        return crest_toe_figure(crest_df, toe_df, load_df, comments_df)

    return dash.no_update


def crest_toe_figure(crest_data, toe_data, load, annotations):

    # Convert date data to date first format
    crest_data["Date"] = pd.to_datetime(crest_data["Date"], dayfirst=True)
    load["Date"] = pd.to_datetime(load["Date"], dayfirst=True)
    toe_data["Date"] = pd.to_datetime(toe_data["Date"], dayfirst=True)

    # re-index dataframe
    crest_data = crest_data.set_index("Date").sort_index()
    load = load.set_index("Date").sort_index()
    toe_data = toe_data.set_index("Date").sort_index()

    min_max_dh = (math.inf, -math.inf, pd.Timestamp.max, pd.Timestamp.min)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[90, 50],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    for identity in crest_data["Instrument_ID"].unique():

        df = crest_data[(crest_data["Instrument_ID"] == identity)]

        df["dh"] = df["Movement"].diff() / df.index.to_series().diff().astype(
            "timedelta64[D]"
        ).astype("Int64")

        df = df.dropna()

        fig.add_trace(
            go.Scatter(x=df.index, y=df["Movement"], name=identity),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df["dh"], name=identity + "-dh"), row=2, col=1
        )

        min_max_dh_loop = min_max(df, "dh")
        min_max_dh = min_max_in_tuples(min_max_dh, min_max_dh_loop)

    # Use construction data here
    # get the state of construction data
    fig.add_trace(
        go.Scatter(
            x=load.index,
            y=load["Height"],
            name=load.iloc[0, 0],
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    min_max_crest = min_max(crest_data, "Movement")

    toe_data = toe_data.dropna()

    for identity in toe_data["Instrument_ID"].unique():

        df = toe_data[(toe_data["Instrument_ID"] == identity)]

        df["dh"] = df["Movement"].diff() / df.index.to_series().diff().astype(
            "timedelta64[D]"
        ).astype("Int64")

        df = df.dropna()

        fig.add_trace(
            go.Scatter(x=df.index, y=df["Movement"], name=identity),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df["dh"], name=identity + "-dh"), row=2, col=1
        )

        min_max_dh_loop = min_max(df, "dh")
        min_max_dh = min_max_in_tuples(min_max_dh, min_max_dh_loop)

    min_max_df_toe = min_max(toe_data, "Movement")
    min_max_df = min_max_in_tuples(min_max_crest, min_max_df_toe)

    min_max_load = min_max(load, "Height")
    min_max_tuples = min_max_in_tuples(min_max_df, min_max_load)

    y_range = min_max_df[1] - min_max_df[0]
    y_add = (10 * y_range) / 100

    x_range = (min_max_tuples[3] - min_max_tuples[2]).days
    x_add = (2 * x_range) / 100
    x_add = timedelta(days=x_add)

    add_annotations(fig, annotations, x_add, y_add, min_max_df)

    fig.update_layout(
        height=650,
        xaxis2_title="Time",
        yaxis1_title="Settlement (mm)",
        yaxis2_title="Construction height (m)",
        yaxis3_title="dh (mm/day)",
    )

    return fig


@app.callback(
    Output("porewater-graph", "figure"),
    Input(Id.Tabs.Id, "active_tab"),
    State(Id.upload.PiezoData, "data"),
    State(Id.upload.ConstructionData, "data"),
    State(Id.upload.AnnotationsData, "data"),
    State("water_weight", "value"),
    State("water_level", "value"),
    State("soil_weight", "value"),
    State("fill_weight", "value"),
    State("ratio_su_s", "value"),
)
def water_pressure_graph(
    tab_name,
    piezo_data,
    load,
    annotations,
    water_weight,
    water_level,
    soil_weight,
    fill_weight,
    ratio,
):
    if tab_name == Id.Tabs.Visualizations.porewater:
        # Convert dictionary to a Pandas data frame object before processing
        piezo_df = pd.DataFrame.from_dict(piezo_data["data"])
        load_df = pd.DataFrame.from_dict(load["data"])
        comm_df = pd.DataFrame.from_dict(annotations["data"])

        # Convert date data to date first format
        piezo_df["Date"] = pd.to_datetime(piezo_df["Date"], dayfirst=True)
        load_df["Date"] = pd.to_datetime(load_df["Date"], dayfirst=True)

        # re-index dataframe
        piezo_df = piezo_df.set_index("Date").sort_index()
        load_df = load_df.set_index("Date").sort_index()

        return water_pressure_figure(
            piezo_df,
            load_df,
            comm_df,
            water_weight,
            water_level,
            soil_weight,
            fill_weight,
            ratio,
        )

    return dash.no_update


def water_pressure_figure(
    piezo_data,
    load,
    comments_data,
    water_weight,
    water_level,
    soil_weight,
    fill_weight,
    ratio,
):

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[60, 30],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    min_max_dh = (math.inf, -math.inf, pd.Timestamp.max, pd.Timestamp.min)

    # Implement a way to dynamically generate UI for Piezo instrument IDs.
    # Each unique instrument will have it's own text input where user can
    # provide anything ranging from -999 to 0. By default is set to 0
    # UI can be a tabular form or a dropdown with a text value next to it

    # for Proof of Concept, it's assumed that we only have one value - VWP7
    for id_du in piezo_data["Instrument_ID"].unique():
        if id_du == "VWP7":

            df = piezo_data[(piezo_data["Instrument_ID"] == id_du)]
            df = pd.merge(df, load, how="left", on=[df.index, load.index])
            df = df.reset_index()
            df = df.set_index("key_0").sort_index()

            df["pp"] = df["Piezometric Level"] * water_weight
            df["epp"] = df["pp"] - ((abs(0 - water_level)) / (1 / water_weight))
            fig.add_trace(go.Scatter(x=df.index, y=df["epp"], name=id_du))
            min_max_dh_loop = min_max(df, "epp")
            min_max_dh = min_max_in_tuples(min_max_dh, min_max_dh_loop)

            # Soil weight must be multiplied by the value of Water instrument value
            # set by the user. for now it is set to '0' by default
            df["sv"] = (soil_weight * 0) + (fill_weight * df["Height"])
            df["svp"] = df["sv"] - df["pp"]
            df["su"] = ratio * df["svp"]
            df["fos"] = (5.14 * df["su"]) / (fill_weight * df["Height"])
            fig.add_trace(
                go.Scatter(x=df.index, y=df["fos"], name=id_du + "-FoS"), row=2, col=1
            )

    fig.add_trace(
        go.Scatter(x=load.index, y=load["Height"], name=load.iloc[0, 0]),
        secondary_y=True,
    )

    min_max_load = min_max(load, "Height")
    min_max_tuples = min_max_in_tuples(min_max_dh, min_max_load)

    y_range = min_max_dh[1] - min_max_dh[0]
    y_add = (10 * y_range) / 100

    x_range = (min_max_tuples[3] - min_max_tuples[2]).days
    x_add = (2 * x_range) / 100
    x_add = timedelta(days=x_add)

    #
    add_annotations(fig, comments_data, x_add, y_add, min_max_dh)

    # Provide the necessary size and labelling for the graph
    fig.update_layout(
        height=650,
        xaxis2_title="Time",
        yaxis1_title="Excess Pore Pressure (kPa)",
        yaxis2_title="Construction height (m)",
        yaxis3_title="FoS",
    )

    return fig


def add_annotations(figure, comments_data, x_add, y_add, min_max):

    # min_max = (math.inf, -math.inf, pd.Timestamp.max, pd.Timestamp.min)

    try:

        for index, row in comments_data.iterrows():

            xs = pd.to_datetime(row[0], infer_datetime_format=True)
            xe = pd.to_datetime(row[0], infer_datetime_format=True)

            figure.add_trace(
                go.Scatter(
                    x=[xs, xe],
                    y=[min_max[0], min_max[1]],
                    name=row[1],
                    opacity=0.5,
                    showlegend=False,
                    line=dict(color="grey", dash="dash"),
                ),
                row=1,
                col=1,
            )

            figure.add_annotation(
                text=row[1],
                x=xs - x_add,
                y=min_max[0] + y_add,
                showarrow=False,
                textangle=-90,
                opacity=0.5,
                font=dict(color="grey"),
            )

    except Exception as error:
        print(error)
        pass


@app.callback(
    Output("guo-chu-graph", "figure"),
    Output(Id.Tabs.Guo.table, "data"),
    Input(Id.Tabs.Id, "active_tab"),
    State(Id.upload.SettlementData, "data"),
)
def guo_chu_figure(tab_name, settlement):

    if tab_name == Id.Tabs.Visualizations.guo:
        # Convert dictionary to a Pandas data frame object before processing
        settlement_df = pd.DataFrame.from_dict(settlement["data"])
        settlement_df["Date"] = pd.to_datetime(settlement_df["Date"], dayfirst=True)
        settlement_df = settlement_df.set_index("Date").sort_index()

        fig = go.Figure()
        table_stats = []

        for guo_id in settlement_df["Instrument_ID"].unique():
            df = settlement_df[(settlement_df["Instrument_ID"] == guo_id)]
            df = df.resample("D").median()

            df["pk"] = abs(df["Settlement"]) ** (5 / 3)
            df["pk+1"] = abs(df["Settlement"].shift(-1)) ** (5 / 3)

            df = df.dropna()

            df_current = df.iloc[-1]["Settlement"]
            fig.add_trace(
                go.Scatter(x=df["pk+1"], y=df["pk"], name=guo_id, mode="markers")
            )

            # Linear Regression
            # values converts it into a numpy array
            X = df[:]["pk+1"].values.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(
                X, df["pk"], random_state=0
            )

            linear_regressor = LinearRegression()
            linear_regressor.fit(X_train, y_train)
            x_range = np.linspace(X.min(), X.max())
            y_range = linear_regressor.predict(x_range.reshape(-1, 1))

            slope = linear_regressor.coef_[0]
            intercept = linear_regressor.intercept_
            table_stats.append(
                {
                    "instrument": guo_id,
                    "gradient": slope,
                    "interceptor": round(intercept, 2),
                    "ultimate_settlement": round((intercept / (1 - slope)) ** 0.6, 2),
                    "diff_to_current": round(
                        (intercept / (1 - slope)) ** 0.6 - abs(df_current), 2
                    ),
                }
            )

            fig.add_trace(
                go.Scatter(x=x_range, y=y_range, name=guo_id + "- Predicted line")
            )

        fig.update_layout(height=600)
        fig.update_xaxes(title_text="S")
        fig.update_yaxes(title_text="Sn+1")

        return fig, table_stats

    return dash.no_update


@app.callback(
    Output("asaoka-graph", "figure"),
    Output(Id.Tabs.Asaoka.table, "data"),
    Input(Id.Tabs.Id, "active_tab"),
    State(Id.upload.SettlementData, "data"),
)
def asaoka_figure(tab_name, settlement):

    if tab_name == Id.Tabs.Visualizations.asaoka:
        # Convert dictionary to a Pandas data frame object before processing
        settlement_df = pd.DataFrame.from_dict(settlement["data"])
        settlement_df["Date"] = pd.to_datetime(settlement_df["Date"], dayfirst=True)
        settlement_df = settlement_df.set_index("Date").sort_index()

        fig = go.Figure()
        table_stats = []

        for asaoka_id in settlement_df["Instrument_ID"].unique():
            df = settlement_df[(settlement_df["Instrument_ID"] == asaoka_id)]
            df = df.resample("D").median()

            df["Settlement"] = abs(df["Settlement"])
            df["pk+1"] = abs(df["Settlement"].shift(-1))

            df = df.dropna()

            df_current = df.iloc[-1]["Settlement"]
            fig.add_trace(
                go.Scatter(
                    x=df["pk+1"], y=df["Settlement"], name=asaoka_id, mode="markers"
                )
            )

            # Linear Regression
            # values converts it into a numpy array
            X = df[:]["pk+1"].values.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(
                X, df["Settlement"], random_state=0
            )

            linear_regressor = LinearRegression()
            linear_regressor.fit(X_train, y_train)
            x_range = np.linspace(X.min(), X.max())
            y_range = linear_regressor.predict(x_range.reshape(-1, 1))

            slope = linear_regressor.coef_[0]
            intercept = linear_regressor.intercept_
            table_stats.append(
                {
                    "instrument": asaoka_id,
                    "gradient": slope,
                    "interceptor": round(intercept, 2),
                    "ultimate_settlement": round(intercept / (1 - slope), 2),
                    "diff_to_current": round(
                        (intercept / (1 - slope)) - abs(df_current), 2
                    ),
                }
            )

            fig.add_trace(
                go.Scatter(x=x_range, y=y_range, name=asaoka_id + "- Predicted line")
            )

        fig.update_layout(height=600)
        fig.update_xaxes(title_text="S")
        fig.update_yaxes(title_text="Sn+1")

        return fig, table_stats

    return dash.no_update


def min_max(df, col):

    mini = df[col].min()
    maxi = df[col].max()

    mini_d = df.index.min()
    maxi_d = df.index.max()

    return (mini, maxi, mini_d, maxi_d)


def min_max_in_tuples(t1, t2):

    mini = min(t1[0], t2[0])
    maxi = max(t1[1], t2[1])
    mini_d = min(t1[2], t2[2])
    maxi_d = max(t1[3], t2[3])

    return (mini, maxi, mini_d, maxi_d)


def get_config(data, name):
    return {
        "id": data[0][name],
        "data": data[2][name],
        "date": data[1][name],
    }


if __name__ == "__main__":
    app.run_server(debug=True)
