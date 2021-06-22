from pathlib import Path
import json
import zipfile

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        # This will be filled in worker_init_fn
        self.sdf_archive = None
        self.df_archive = None

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = self.get_shape_sdf(sdf_id)
        target_df = self.get_shape_df(df_id)

        # TODO Apply truncation to sdf and df
        input_sdf = np.clip(input_sdf, a_min=-self.truncation_distance, a_max=self.truncation_distance)
        target_df = np.clip(target_df, a_min=0, a_max=self.truncation_distance)
        # TODO Stack (distances, sdf sign) for the input sdf
        input_sdf = np.stack([np.abs(input_sdf), np.sign(input_sdf)])  # (distances, sdf sign)
        # TODO Log-scale target df
        target_df = np.log(np.add(target_df, 1.0))
        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    def worker_init_fn(self, worker_id):
        self.sdf_archive = zipfile.ZipFile('exercise_3/data/shapenet_dim32_sdf.zip', 'r')
        self.df_archive = zipfile.ZipFile('exercise_3/data/shapenet_dim32_df.zip', 'r')

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        batch["input_sdf"] = batch["input_sdf"].to(device)
        batch["target_df"] = batch["target_df"].to(device)

    def get_shape_sdf(self, shapenet_id):
        data = self.sdf_archive.read(f'shapenet_dim32_sdf/{shapenet_id}.sdf')
        sdf = None
        # TODO implement sdf data loading
        new_pth = "exercise_3/data/shapenet_dim32_sdf/"+shapenet_id + ".sdf"
        dims = np.fromfile(new_pth,dtype=np.uint64,count=3)
        new_mat = np.fromfile(new_pth,dtype=np.float32,offset = 24)
        new_mat = new_mat.reshape((dims[0],dims[1],dims[2]))
        sdf = new_mat

    def get_shape_df(self, shapenet_id):
        data = self.df_archive.read(f'shapenet_dim32_df/{shapenet_id}.df')
        df = None
        # TODO implement df data loading
        df = new_mat
        return df
