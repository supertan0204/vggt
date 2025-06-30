# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np

from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset
from training.data.dynamic_dataloader import DynamicDistributedSampler, DynamicBatchSampler


SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]


class Co3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        CO3D_DIR: str = None,
        CO3D_ANNOTATION_DIR: str = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
    ):
        """
        Initialize the Co3dDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            CO3D_DIR (str): Directory path to CO3D data.
            CO3D_ANNOTATION_DIR (str): Directory path to CO3D annotations.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If CO3D_DIR or CO3D_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.seed = common_conf.seed
        self.duplicate_img = common_conf.duplicate_img if hasattr(common_conf, 'duplicate_img') else False

        if CO3D_DIR is None or CO3D_ANNOTATION_DIR is None:
            raise ValueError("Both CO3D_DIR and CO3D_ANNOTATION_DIR must be specified.")

        category = sorted(SEEN_CATEGORIES)

        if split == "train":
            split_name_list = ["train"]
            self.len_train = len_train
        elif split == "test":
            split_name_list = ["test"]
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = [] # set any invalid sequence names here


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"CO3D_DIR is {CO3D_DIR}")

        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR

        total_frame_num = 0

        for c in category:
            for _ in split_name_list:
                # annotation_file = osp.join(
                #     self.CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz"
                # )
                annotation_file = osp.join(
                    self.CO3D_ANNOTATION_DIR, c, "sequence_annotations.jgz"
                )
                frame_annotation_file = osp.join(self.CO3D_ANNOTATION_DIR, c, "frame_annotations.jgz")

                try:
                    with gzip.open(annotation_file, "r") as fin:
                        annotation = json.loads(fin.read())
                except FileNotFoundError:
                    logging.error(f"Annotation file not found: {annotation_file}")
                    continue
                
                try:
                    with gzip.open(frame_annotation_file, "r") as fin:
                        frame_annotation = json.loads(fin.read())
                except FileNotFoundError:
                    logging.error(f"Frame annotation file not found: {frame_annotation_file}")
                    continue
                # logging.info(f"annotaion length: {len(annotation)}")
                # logging.info(f"frame_annotation length: {len(frame_annotation)}")
                # count = 0
                # for item in annotation:
                #     count += 1 
                #     print(f"sequence name: {item['sequence_name']}, frame number: {item['frame_number']}, file name: {item['image']['path']}, count: {count}")
                # print(annotation[0])
                # print(frame_annotation[0])
                
                for item in annotation:
                    seq_name = item["sequence_name"]
                    seq_path = osp.join(self.CO3D_DIR, c, seq_name)
                    if not osp.exists(seq_path):
                        logging.warning(f"Sequence path does not exist: {seq_path}")
                        continue
                    img_path = osp.join(seq_path, "images")
                    frame_num = len([f for f in os.listdir(img_path) if osp.isfile(os.path.join(img_path, f))])
                    self.data_store[seq_name] = seq_path
                    total_frame_num += frame_num
                    # print(f"sequence path: {seq_path}, frame number: {num_frame}")
                
                # for seq_name, seq_data in annotation.items():
                #     if len(seq_data) < min_num_images:
                #         continue
                #     if seq_name in self.invalid_sequence:
                #         continue
                #     total_frame_num += len(seq_data)
                #     self.data_store[seq_name] = seq_data

        self.sequence_list = list(self.data_store.keys()) # the list of all sequences, like: ['110_13051_23361', '189_20393_38136', ...]
        self.sequence_list_len = len(self.sequence_list) 
        self.total_frame_num = total_frame_num
        # print(self.sequence_list) 
        # print(self.sequence_list_len)
        # print(self.total_frame_num)
        # print(self.data_store[self.sequence_list[0]]) # data_store is a dict, the key is the sequence name, the value is the path to the sequence
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Co3D Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Co3D Data dataset length: {len(self)}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
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
        # logging.info(f"seq_index: {seq_index}, img_per_seq: {img_per_seq}, seq_name: {seq_name}, ids: {ids}, aspect_ratio: {aspect_ratio}")
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        seq_file = self.data_store[seq_name]
        file_list = os.listdir(osp.join(seq_file, "images"))
        # logging.info(f"seq_file: {seq_file}, file_list: {file_list}")
        num_files = len([f for f in file_list if osp.isfile(os.path.join(seq_file, "images", f))])
        # logging.info(f"num_files: {num_files} in sequence {seq_name},/home/ubuntu/nvme/xiyang/vggt/co3d/cake/374_42274_84517

        if ids is None:
            ids = np.random.choice(
                num_files, img_per_seq, replace=self.duplicate_img
            ) # select some random images from a given sequence
            # ids = np.random.choice(
            #     len(seq_file), img_per_seq, replace=self.duplicate_img
            # )
            # # annos = [metadata[i] for i in ids]
        # logging.info(f"ids: {ids}")
        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        # cam_points = []
        # world_points = []
        # point_masks = []
        # extrinsics = []
        # intrinsics = []
        image_paths = []
        original_sizes = []

        # for anno in annos:
        for id in ids:
            # filepath = anno["filepath"]

            # image_path = osp.join(self.CO3D_DIR, filepath)
            # logging.info(f"seq_file: {seq_file}, file_list[id]: {file_list[id]}")
            image_path = osp.join(seq_file, "images", file_list[id])
            image = read_image_cv2(image_path)
            # logging.info(f"Successfully read image from {image_path}")

            if self.load_depth:
                depth_path = image_path.replace("/images", "/depths") + ".geometric.png"
                logging.info(f"Loading depth map from {depth_path}")
                depth_map = read_depth(depth_path, 1.0)

                if self.mask_depth:
                    mvs_mask_path = image_path.replace(
                        "/images", "/depth_masks"
                    ).replace(".jpg", ".png")
                    mvs_mask = cv2.imread(mvs_mask_path, cv2.IMREAD_GRAYSCALE) > 128
                    depth_map[~mvs_mask] = 0

                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=98
                )
            else:
                depth_map = None

            original_size = np.array(image.shape[:2])
            # extri_opencv = np.array(anno["extri"])
            # intri_opencv = np.array(anno["intri"])
            
            (
                image,
                depth_map,    
            ) = self.process_one_image_simple(
                image, 
                depth_map,
                original_size,
                target_image_shape,
                rescale_aug=False,
                safe_bound=4
            )
            # (
            #     image,
            #     depth_map,
            #     # extri_opencv,
            #     # intri_opencv,
            #     world_coords_points,
            #     cam_coords_points,
            #     point_mask,
            #     _,
            # ) = self.process_one_image(
            #     image,
            #     depth_map,
            #     # extri_opencv,
            #     # intri_opencv,
            #     original_size,
            #     target_image_shape,
            #     # filepath=filepath,
            #     filepath=image_path
            # )
            # print(image)
            images.append(image)
            depths.append(depth_map)
            # extrinsics.append(extri_opencv)
            # intrinsics.append(intri_opencv)
            # cam_points.append(cam_coords_points)
            # world_points.append(world_coords_points)
            # point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        set_name = "co3d"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            # "frame_num": len(extrinsics),
            "frame_num": len(images),
            "images": images,
            # "depths": depths,
            # "extrinsics": extrinsics,
            # "intrinsics": intrinsics,
            # "cam_points": cam_points,
            # "world_points": world_points,
            # "point_masks": point_masks,
            # "original_sizes": original_sizes,
        }
        # logging.info(f"Batch created for {len(images)} images with size {images[0].shape}.")
        return batch


