import os

import objaverse

OBJAVERSE_MODEL_URL = 'https://huggingface.co/datasets/allenai/objaverse/resolve/main'
OBJAVERSE_OBJECT_PATHS = objaverse._load_object_paths()


def get_objaverse_path_from_uid(uid):
    return OBJAVERSE_OBJECT_PATHS[uid]


def get_objaverse_url_from_uid(uid):
    return os.path.join(OBJAVERSE_MODEL_URL, OBJAVERSE_OBJECT_PATHS[uid])
