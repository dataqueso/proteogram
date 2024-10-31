from .constants import RESIDUE_LIST, UNKNOWN_RESIDUE
import yaml


def get_3letter_res_name(res_code):
    lookup = dict(RESIDUE_LIST)
    return lookup.get(res_code, UNKNOWN_RESIDUE[1])

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)