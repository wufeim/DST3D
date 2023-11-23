import os


def get_pkg_root():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, ".."))
    return root


def get_project_root():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, "..", ".."))
    return root


def get_abs_path(path):
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path
