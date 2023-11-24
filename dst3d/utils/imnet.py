import csv
import io
import os
import requests

import numpy as np
import objaverse

from .imnet_classes import imnet_classes

_imnet_synset_to_idx = [x.split(' ')[0] for x in imnet_classes]
_imnet_synset_to_class = {x.split(' ')[0]: ' '.join(x.split(' ')[1:]) for x in imnet_classes}


def call_objaverse_api(model_id, fields=None):
    # Some useful fields: 'uri', 'uid', 'name', 'tags', 'categories', 'description'
    anno = objaverse.load_annotations([model_id])[model_id]
    if fields is not None:
        anno = {k: anno[k] for k in fields}
    return anno


def call_objaverse_api(model_id, fields=None):
    # Some interesting fields: 'uri', 'uid', 'name', 'tags', 'categories', 'description'
    anno = objaverse.load_annotations([model_id])[model_id]
    if fields is not None:
        anno = {k: anno[k] for k in fields}
    return anno


def call_shapenet_api(model_id, fields=None):
    url = 'https://shapenet.org/solr/models3d/select?q=datasets%3AShapeNetCore+AND'
    url += f'+id%3A{model_id}'
    url += '&rows=100000'
    if fields is not None:
        url += f'&fl={"%2C".join(fields)}'
    url += '&wt=csv&indent=true'
    with requests.Session() as sess:
        download = sess.get(url)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
    info = {}
    for i, f in enumerate(fields):
        info[f] = my_list[1][i]
    return info


def _get_keywords_objaverse(model_id):
    info = call_objaverse_api(model_id)
    if len(info['categories']) > 0:
        return [x['name'] for x in info['tags']] + [info['categories'][0]['name']]
    else:
        return [x['name'] for x in info['tags']]


def _get_keywords_shapenet(model_id):
    info = call_shapenet_api(model_id, fields=['fullId', 'wnlemmas', 'name', 'description'])
    return info['wnlemmas'].split(',') + [info['name']]


def get_keywords(model_id, model_src='shapenet'):
    if model_src == 'shapenet':
        return _get_keywords_shapenet(model_id)
    elif model_src == 'objaverse':
        return _get_keywords_objaverse(model_id)
    else:
        raise ValueError(f'Unknown model source {model_src}')


def imnet_synset_to_idx(synset):
    return _imnet_synset_to_idx.index(synset)


def imnet_synset_to_class(synset):
    return _imnet_synset_to_class[synset]


def imnet_idx_to_synset(idx):
    return _imnet_synset_to_idx[idx]


def imnet_get_all_synsets():
    return _imnet_synset_to_idx
