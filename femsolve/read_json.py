import numpy as np
import json


def read_json(path_settings):
    """
    Read settings from json file
    """
    try:
        with open(path_settings, 'r', encoding='utf-8') as f:
            datas = json.load(f)

            # read geometry
            geometry_set = []
            for key, value in datas["geometry"].items():
                geometry_set.append(value)

            # read mesh settings
            mesh_set = []
            for key, value in datas["mesh"].items():
                mesh_set.append(value)

            # read boundaries
            boundaries_set = []
            for key, value in datas["boundaries"].items():
                boundaries_set.append(value)

            # read material properties
            properties_set = []
            for key, value in datas["properties"].items():
                properties_set.append(value)

            # read solver settings
            solver_set = []
            for key, value in datas["solver"].items():
                solver_set.append(value)

            # reading plotter settings
            plotter_set = []
            for key, value in datas["plotter"].items():
                plotter_set.append(value)

    except json.JSONDecodeError as err:
        print("Wrong json format", err)
    except FileNotFoundError:
        print("No such file, please check the path")

    return geometry_set, mesh_set, boundaries_set, properties_set, solver_set, plotter_set
