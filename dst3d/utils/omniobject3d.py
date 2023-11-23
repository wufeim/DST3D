import os


def get_omniobject3d_path_map(omniobject3d_path):
    omniobject3d_path = os.path.join(omniobject3d_path, 'raw', 'raw_scans')
    cates = [x for x in os.listdir(omniobject3d_path) if not x.startswith('.') and 'tar.gz' not in x]
    mapping = {}
    for c in cates:
        models = [x for x in os.listdir(os.path.join(omniobject3d_path, c)) if x.startswith(c)]
        for m in models:
            mapping[f'omniobject3d_{m}'] = os.path.join('raw', 'raw_scans', c, m, 'Scan', 'Scan.obj')
    return mapping
