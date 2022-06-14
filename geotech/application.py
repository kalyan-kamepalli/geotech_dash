import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from geotech.app import app


if __name__ == "__main__":
    app.run_server(debug=True)
