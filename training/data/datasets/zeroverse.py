# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import json
import logging
import random

from sympy.ntheory.continued_fraction import continued_fraction
import numpy as np

from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset


class ZeroVerseDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ZEROVERSE_DIR: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the ZeroVerse dataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            ZEROVERSE_DIR (str): Directory path to ZeroVerse data.
            ZEROVERSE_ANNOTATION_DIR (str): Directory path to ZeroVerse annotations.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If ZEROVERSE_DIR or ZEROVERSE_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if ZEROVERSE_DIR is None:
            raise ValueError("ZEROVERSE_DIR must be specified.")

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = []  # set any invalid sequence names here

        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"ZEROVERSE_DIR is {ZEROVERSE_DIR}")

        self.ZEROVERSE_DIR = ZEROVERSE_DIR

        total_frame_num = 0

        # Basic framework for ZeroVerse data loading
        # This will populate self.data_store with sequence data
        # and calculate total_frame_num
        
        # Get all available sequence directories directly from ZEROVERSE_DIR
        all_sequences = []
        if osp.exists(self.ZEROVERSE_DIR):
            for item in os.listdir(self.ZEROVERSE_DIR):
                item_path = osp.join(self.ZEROVERSE_DIR, item)
                if osp.isdir(item_path):
                    all_sequences.append(item)
        
        if not all_sequences:
            raise ValueError(f"No sequence directories found in {self.ZEROVERSE_DIR}")
        
        # Sort sequences for consistent splitting
        all_sequences.sort()
        total_sequences = len(all_sequences)
        
        # Split sequences 80/20 for train/test
        if split == "train":
            # Use 80% of sequences for training
            num_train = int(total_sequences * 0.8)
            selected_sequences = all_sequences[:num_train]
            logging.info(f"Training split: using {len(selected_sequences)} out of {total_sequences} sequences (80%)")
        else:  # split == "test"
            # Use remaining 20% of sequences for testing
            num_train = int(total_sequences * 0.8)
            selected_sequences = all_sequences[num_train:]
            logging.info(f"Testing split: using {len(selected_sequences)} out of {total_sequences} sequences (20%)")
        
        # Iterate through selected sequences
        for seq_name in selected_sequences:
            seq_path = osp.join(self.ZEROVERSE_DIR, seq_name)
            if not osp.isdir(seq_path):
                logging.warning(f"Sequence path didn't found: {seq_path}, skipping...")
                continue
                    
            # Load annotation
            annotation_file = osp.join(seq_path, "views", "opencv_cameras.json")
            
            try:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
                logging.warning(f"Failed to load annotation file {annotation_file}: {e}, skipping...")
                continue
            
            # Extract frame information from the annotation data
            frames = []
            
            # Check if annotation_data has the expected structure
            if isinstance(annotation_data, dict) and 'frames' in annotation_data:
                frames = annotation_data['frames']
            elif isinstance(annotation_data, list):
                frames = annotation_data
            else:
                logging.warning(f"Unexpected annotation format in {annotation_file}")
                continue
            
            # Filter frames based on minimum image requirement
            if len(frames) >= min_num_images:
                self.data_store[seq_name] = frames
                total_frame_num += len(frames)
            else:
                continue

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: ZeroVerse Data size: {self.sequence_list_len}")
        logging.info(f"{status}: ZeroVerse Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index=None,
        seq_name=None,
        img_per_seq=None,
        ids=None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        if seq_name is None:
            if seq_index is None:
                seq_index = 0
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]

        # Default to using all available frames if no specific ids provided
        if ids is None:
            # Use a reasonable default number of images per sequence
            ids = np.random.choice(
                len(metadata), img_per_seq, replace=self.allow_duplicate_img
            )

        annos = [metadata[i] for i in ids]
        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []
        
        for anno in annos:
            file_path = anno["file_path"]
            
            image_path = osp.join(self.ZEROVERSE_DIR, seq_name, "views", file_path)
            image = read_image_cv2(image_path)
            depth_map = None
            
            
            original_size = np.array(image.shape[:2])
            extri_opencv = np.array(anno["w2c"])
            
            # Extract camera parameters from annotation
            w = anno.get("w", 0)
            h = anno.get("h", 0)
            fx = anno.get("fx", 0.0)
            fy = anno.get("fy", 0.0)
            cx = anno.get("cx", 0.0)
            cy = anno.get("cy", 0.0)
            
            # Calculate scale ratio to adjust intrinsics based on actual image size
            # This follows the same logic as dl3dv.py
            scale_ratio = max(w / original_size[1], h / original_size[0])
            fx_current = fx / scale_ratio
            fy_current = fy / scale_ratio
            cx_current = cx / scale_ratio
            cy_current = cy / scale_ratio
            
            # Generate intrinsic matrix (OpenCV convention)
            intri_opencv = np.array(
                [
                    [fx_current, 0, cx_current],
                    [0, fy_current, cy_current],
                    [0, 0, 1]
                ],
                dtype=np.float32
            )
            
            # Process the image using the base dataset method
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
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_path,
            )
            # Create an all-valid boolean mask since point_mask is not needed for ZeroVerse
            point_mask = np.ones(original_size, dtype=bool)
            
            # Append processed data to lists
            images.append(image)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)
            
        # We do not need world_points, depth_maps
        world_points = None
        depths = None
        cam_points = None
        set_name = "zeroverse"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "image_paths": image_paths,
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        
        return batch
