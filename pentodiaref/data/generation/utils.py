import json
import os

from golmi.server.grid import GridConfig
from golmi.server.state import State
from tqdm import tqdm


def load_sent_types(data_dir, filename="sent_types"):
    file_path = os.path.join(data_dir, filename + ".json")
    with open(file_path, "r") as f:
        return json.load(f)


def load_json_states(data_dir, file_name):
    file_path = os.path.join(data_dir, f"{file_name}.states")
    grid_config = GridConfig.load(data_dir)
    print("Load states from", file_path)
    print("Warning: This method might be very slow!")
    states = []
    with open(file_path, "r") as f:
        for line in tqdm(f.readlines(), position=0, leave=True):
            states.append(State.from_dict(json.loads(line), grid_config=grid_config))
    return states


def add_gid_and_split_name(samples, gid_start, split_name):
    gid = gid_start
    for sample in samples:
        sample.global_id = gid
        sample.split_name = split_name
        gid += 1
    return gid
