from .constants import RESIDUE_LIST, UNKNOWN_RESIDUE


def get_3letter_res_name(res_code):
    lookup = dict(RESIDUE_LIST)
    return lookup.get(res_code, UNKNOWN_RESIDUE[1])