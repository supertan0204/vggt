import os
from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset
from typing import List, Optional
import random
import json
import logging

DIVIDES = [
    "1K",
    "2K",
    "3K",
    "4K",
    "5K",
    "6K",
    "7K",
    "8K",
    "9K",
    "10K",
]



class DL3DVDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        stage: str = "train",
        DL3DV_ROOT: str = None,
        divides: List[str] = [],
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the DL3DV dataset.
        """
        super().__init__(common_conf=common_conf)
        
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        if DL3DV_ROOT is None:
            raise ValueError("DL3DV_DIR_ROOT must be specified.")
        if not set(divides).issubset(set(DIVIDES)):
            raise ValueError(f"'divide' must be a subset of DIVIDE: {DIVIDES}, but got {divides}")
        
        self.DL3DV_ROOT = DL3DV_ROOT
        
        if stage == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif stage == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {stage}")    
        
        self.metadatas = {}
        for divide in divides:
            chunk_path = os.path.join(DL3DV_ROOT, divide)
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Chunk path does not exist: {chunk_path}")
            for folder_hash in os.listdir(chunk_path):
                folder_path = os.path.join(chunk_path, folder_hash)
                annotation_file = os.path.join(folder_path, "transforms.json")
                try:
                    with open(annotation_file, 'r') as f:
                        annotations = json.load(f)
                except FileNotFoundError:
                    logging.error(f"Annotation file does not exist: {annotation_file}")
                    continue
                
                self.metadatas[folder_path] = annotations
                
            self.hash_path_list = list(self.metadatas.keys())
            self.hash_list_len = len(self.hash_path_list)
    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_path: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ):
        if self.inside_random:
            seq_index = random.randint(0, self.hash_list_len - 1)
        if seq_path is None:
            seq_path = self.hash_path_list[seq_index]
        
        metadata = self.metadatas[seq_path]
        if ids is None:
            ids = np.random.choice(
                len(metadata), img_per_seq, replace=self.allow_duplicate_img
            )
        
        # extract ids of frames
        frame_annos = [metadata['frames'][i] for i in ids]
        target_image_shape = self.get_target_shape(aspect_ratio)
        
        # extract camera intrinsics for a certain frame
        fl_x = metadata.get("fl_x", 0.0)
        fl_y = metadata.get("fl_y", 0.0)
        cx = metadata.get("cx", 0.0)
        cy = metadata.get("cy", 0.0)
        k1 = metadata.get("k1", 0.0)
        k2 = metadata.get("k2", 0.0)
        p1 = metadata.get("p1", 0.0)
        p2 = metadata.get("p2", 0.0)
        
        missing_frames = []
        images = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        image_paths = []
        
        for frame_anno in frame_annos:
            frame_path = frame_anno.get("file_path", "")
            actual_frame_path = os.path.join(self.DL3DV_ROOT, seq_path, frame_path)
            if not os.path.exists(actual_frame_path):
                actual_frame_path = actual_frame_path.replace("images", "images_8")
                if not os.path.exists(actual_frame_path):
                    missing_frames.append(actual_frame_path)
                    continue
            
            # read image and process it
            image = read_image_cv2(actual_frame_path)
            original_size = np.array(image.shape[:2])
            extri_opencv = np.array(frame_anno.get("transform_matrix", []), dtype=np.float32)
            intri_opencv = np.array(
                            [
                                [fl_x, 0, cx],
                                [0, fl_y, cy],
                                [0, 0, 1]
                            ], 
                            dtype=np.float32
                            )
            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                None,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=actual_frame_path,
            )
            images.append(image)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            original_sizes.append(original_size)
            image_paths.append(actual_frame_path)
                
        set_name = "dl3dv"
        batch = {
            "seq_name": set_name + "_" + seq_path,
            "ids": ids,
            "image_paths": image_paths,
            "frame_num": len(extrinsics),
            "images": images,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "original_sizes": original_sizes,
        }
        
        if len(missing_frames) > 0:
            logging.warning(f"Missing frames: {missing_frames}")
        return batch
                
        
        
        
            
            
        
        
        
                        

        