import csv
import os
import requests


def get_shapenet_path_map(shapenet_path):
    path_map = {}
    all_class_ids = os.listdir(shapenet_path)
    for class_id in all_class_ids:
        if class_id == 'taxonomy.json':
            continue
        uids = os.listdir(os.path.join(shapenet_path, class_id))
        for uid in uids:
            path_map[uid] = os.path.join(class_id, uid)
    return path_map
