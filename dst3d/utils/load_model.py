import os

import wget

from .objaverse import OBJAVERSE_OBJECT_PATHS, get_objaverse_url_from_uid
from .omniobject3d import get_omniobject3d_path_map
from .shapenet import get_shapenet_path_map


class ModelLoader:
    def __init__(self, shapenet_path=None, objaverse_path=None, omniobject3d_path=None, toys4k_path=None):
        assert shapenet_path is not None or objaverse_path is not None or omniobject3d_path is not None or toys4k_path is not None, \
            'At least one of the four dataset should be available'
        self.shapenet_path = shapenet_path
        self.objaverse_path = objaverse_path
        self.omniobject3d_path = omniobject3d_path
        self.toys4k_path = toys4k_path

        # Set up ShapeNet
        if self.shapenet_path is not None:
            self.shapenet_path_map = get_shapenet_path_map(self.shapenet_path)

        # Set up Objaverse
        if self.objaverse_path is not None:
            os.makedirs(self.objaverse_path, exist_ok=True)

        # Set up OmniObject3D
        if self.omniobject3d_path is not None:
            self.omniobject3d_path_map = get_omniobject3d_path_map(self.omniobject3d_path)

    def load_model(self, model_id):
        if self.shapenet_path is not None and model_id in self.shapenet_path_map:
            model_path = os.path.join(self.shapenet_path, self.shapenet_path_map[model_id])
            if 'model_normalized.obj' in os.listdir(model_path):
                model_path = os.path.join(model_path, 'model_normalized.obj')
            elif os.path.isfile(os.path.join(model_path, 'models', 'model_normalized.obj')):
                model_path = os.path.join(model_path, 'models', 'model_normalized.obj')
            else:
                model_path = os.path.join(model_path, 'model.obj')  # shapenet v1
        elif self.objaverse_path is not None and model_id in OBJAVERSE_OBJECT_PATHS:
            model_path = os.path.join(self.objaverse_path, f'{model_id}.glb')
            if not os.path.isfile(model_path):
                wget.download(get_objaverse_url_from_uid(model_id), out=self.objaverse_path)
                print('download', model_path, os.path.isfile(model_path))
        elif self.omniobject3d_path is not None and model_id in self.omniobject3d_path_map:
            model_path = os.path.join(self.omniobject3d_path, self.omniobject3d_path_map[model_id])
        elif self.toys4k_path is not None and model_id.startswith('toys4k'):
            model_id = model_id[7:]
            model_path = os.path.join(self.toys4k_path, '_'.join(model_id.split('_')[:-1]), model_id, model_id+'.blend')
        else:
            raise NotImplementedError(f'No model {model_id} found')
        return model_path

    def has_model(self, model_id):
        if self.shapenet_path is not None and model_id in self.shapenet_path_map:
            return True
        elif self.objaverse_path is not None and model_id in OBJAVERSE_OBJECT_PATHS:
            return True
        elif self.omniobject3d_path is not None and model_id in self.omniobject3d_path_map:
            return True
        elif self.toys4k_path is not None and model_id.startswith('toys4k'):
            return True
        else:
            return False
