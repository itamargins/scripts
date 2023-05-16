import os
import sys
import json
import random
import pandas as pd
from uuid import uuid4
import argparse


def load_coco_dataset(data_path):
    with open(data_path) as fp:
        dataset = json.load(fp)

    # print(f"Loaded dataset's image size: w: {dataset['images'][0]['width']}, h: {dataset['images'][0]['height']}")
    # print(f'downscale_factor: {args.image_downscale_factor}\n\n')

    # # downscale:
    # for i in range(len(dataset['images'])):
    #     dataset['images'][i]['width'] /= args.image_downscale_factor
    #     dataset['images'][i]['height'] /= args.image_downscale_factor
    # for i in range(len(dataset['annotations'])):
    #     dataset['annotations'][i]['center'][0] /= args.image_downscale_factor
    #     dataset['annotations'][i]['center'][1] /= args.image_downscale_factor
    #     dataset['annotations'][i]['w_h'][0] /= args.image_downscale_factor
    #     dataset['annotations'][i]['w_h'][1] /= args.image_downscale_factor


    # map image file name -> image id
    dataset['img_to_idx'] = {img['file_name']: img['id'] for img in dataset['images']}

    # map image id -> image object
    dataset['imgs'] = {img['id']: img for img in dataset['images']}

    # map image id -> list of annotation objects
    dataset['img_annotations'] = {img['id']: [] for img in dataset['images']}
    for ann in dataset['annotations']:
        ann_image_id = ann['image_id']
        # if ann_image_id not in dataset['img_annotations']:
        #     dataset['img_annotations'][ann_image_id] = []
        dataset['img_annotations'][ann_image_id].append(ann)

    return dataset


if __name__ == "__main__":
    """
    python coco_train_test_val_split.py --path_to_data <PATH>
    This script takes a json file and randomly allocates images to train/test/val.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_frac', default=0.7,
                        help="what fraction of the data goes to trainset? in [0,1] s.t all fractions sum to 1")
    parser.add_argument('--test_frac', default=0.2,
                        help="what fraction of the data goes to testset? in [0,1] s.t all fractions sum to 1")
    parser.add_argument('--val_frac', default=0.1,
                        help="what fraction of the data goes to validation_set? in [0,1] s.t all fractions sum to 1")
    parser.add_argument('--dataset', help="path to the data json file")
    # parser.add_argument('--image_downscale_factor', help="path to the data json file", type=float, default=1.0)

    args = parser.parse_args()
    TRAIN_FRAC = float(args.train_frac)
    TEST_FRAC = float(args.test_frac)
    VAL_FRAC = float(args.val_frac)
    assert TRAIN_FRAC + TEST_FRAC + VAL_FRAC == 1
    TRAIN_FRAC = float(args.train_frac)
    TEST_FRAC = float(args.test_frac)
    VAL_FRAC = float(args.val_frac)
    dataset_file = args.dataset

    dataset = load_coco_dataset(dataset_file)
    print(f"total num images: {len(dataset['images'])}, total num annotations: {len(dataset['annotations'])}")
    # count TN samples
    print(f"num TN images: {len([dataset['img_annotations'][img] for img in dataset['imgs'] if len(dataset['img_annotations'][img])==0])}")

    train_dataset = dataset.copy()
    test_dataset = dataset.copy()
    val_dataset = dataset.copy()

    image_idx = list(dataset['imgs'].keys())
    random.shuffle(image_idx)

    train_size = round(len(image_idx) * TRAIN_FRAC)
    test_size = round(len(image_idx) * TEST_FRAC)
    train_image_idx = image_idx[:train_size]
    test_image_idx = image_idx[train_size:train_size + test_size]
    val_image_idx = image_idx[train_size + test_size:]

    # TRAIN
    train_dataset['images'] = [train_dataset['imgs'][i] for i in train_image_idx]
    train_dataset['imgs'] = {img['id']: img for img in train_dataset['images']}
    train_dataset['annotations'] = []
    for i in train_image_idx:
        train_dataset['annotations'].extend(train_dataset['img_annotations'][i])

    print("train num images:", len(train_dataset['images']))
    print("train num annotations:", len(train_dataset['annotations']))
    print(f"train num TN images: {len([train_dataset['img_annotations'][img] for img in train_dataset['imgs'] if len(train_dataset['img_annotations'][img])==0])}")

    del train_dataset['img_to_idx']
    del train_dataset['imgs']
    del train_dataset['img_annotations']

    train_dataset_file = dataset_file.replace(".json", "_train.json")
    with open(train_dataset_file, "w") as fp:
        json.dump(train_dataset, fp)

    # TEST
    test_dataset['images'] = [test_dataset['imgs'][i] for i in test_image_idx]
    test_dataset['imgs'] = {img['id']: img for img in test_dataset['images']}
    test_dataset['annotations'] = []
    for i in test_image_idx:
        test_dataset['annotations'].extend(test_dataset['img_annotations'][i])

    print("test num images:", len(test_dataset['images']))
    print("test num annotations:", len(test_dataset['annotations']))
    print(f"test num TN images: {len([test_dataset['img_annotations'][img] for img in test_dataset['imgs'] if len(test_dataset['img_annotations'][img])==0])}")


    del test_dataset['img_to_idx']
    del test_dataset['imgs']
    del test_dataset['img_annotations']

    test_dataset_file = dataset_file.replace(".json", "_test.json")
    with open(test_dataset_file, "w") as fp:
        json.dump(test_dataset, fp)

    # VALIDATION
    val_dataset['images'] = [val_dataset['imgs'][i] for i in val_image_idx]
    val_dataset['imgs'] = {img['id']: img for img in val_dataset['images']}
    val_dataset['annotations'] = []
    for i in val_image_idx:
        val_dataset['annotations'].extend(val_dataset['img_annotations'][i])

    print("val num images:", len(val_dataset['images']))
    print("val num annotations:", len(val_dataset['annotations']))
    print(f"val num TN images: {len([val_dataset['img_annotations'][img] for img in val_dataset['imgs'] if len(val_dataset['img_annotations'][img])==0])}")


    del val_dataset['img_to_idx']
    del val_dataset['imgs']
    del val_dataset['img_annotations']

    val_dataset_file = dataset_file.replace(".json", "_val.json")
    with open(val_dataset_file, "w") as fp:
        json.dump(val_dataset, fp)
