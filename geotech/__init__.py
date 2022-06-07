import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

CSS_COLORS = ["red", "blue", "green", "orange", "magenta", "dark green", "purple"]


class Layout:
    TempData = "div-temp-data"  # hidden div which stores imported csv data
    StoreLocal = "mudmap_saved_files"
    StoreSession = "session_temp_store"
    Reload = "url"
    ColumnData = "col-data"
    ColumnGraph = "col-graph"
    fileselection = "user-file-format"

    class upload:
        annotation = "btn-annotation"
        AnnotationsData = "annotation_data"
        SettlementData = "settlement_data"
        ConstructionData = "construction_data"
        PiezoData = "piezo_data"
        CrestData = "crest_data"
        ToeData = "toe_data"
        settlement = "btn-settlement"
        construction = "btn-construction"
        piezo = "btn-piezo"
        crest = "btn-crest"
        toe = "btn-toe"

    class NavBar:
        BtnHome = "btn-home"
        BtnData = "btn-data"
        BtnViz = "btn-viz"

    class Main:
        NetworkGraph = "svg-graph"
        MainRow = "row-main"
        DotInput = "input-dot"
        DotInputToggle = "input-toggle-dotedit"
        Legend = "table-legend"

    class Tabs:
        Id = "my-tabs"

        class Home:
            Id = "tab-home"

        class UserData:
            Id = "tab-user-data"
            config = "user-config"

        class Visualizations:
            settlement = "settlement-vizs"
            construction = "construction-vizs"
            piezo = "piezo_vizs"
            crest = "crest_vizs"
            asaoka = "asaoka_vizs"
            porewater = "porewater_vizs"
            guo = "guo_vizs"

        class Guo:
            table = "guo-chu-table"

        class Asaoka:
            table = "asaoka-table"
