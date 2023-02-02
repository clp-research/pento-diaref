import os
import random
from typing import Dict, Union

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms, ToTensor, Resize, RandomAffine

from pentodiaref.data.utils import get_tokenizer, load_annotations


def crop_piece(context_image, target_idx, bboxes):
    x, xx, y, yy = bboxes[target_idx]
    height, width, channels = context_image.shape  # probe for sizes
    # crop some additional context (if possible)
    py, pyy = np.clip(y - 5, 0, height), np.clip(yy + 5, 0, height)
    px, pxx = np.clip(x - 5, 0, width), np.clip(xx + 5, 0, width)
    # create a copy of the bbox, so that transforms do not interfere with each other
    target_image = context_image[py:pyy, px:pxx, :].copy()
    rel_size = (abs(x - xx) * abs(y - yy)) / (width * height)
    target_attributes = torch.tensor([x / width, y / height, xx / width, yy / height, rel_size])
    return target_image, target_attributes


class DataMode:

    def __init__(self, mode, subsample_percentage: Union[int, float] = None):
        if mode not in ["only_targets", "default_generation", "sequential_generation",
                        "default_classification", "sequential_classification"]:
            raise Exception("Unknown data mode:", mode)
        self.mode = mode
        self.subsample_percentage = subsample_percentage

    def is_subsample(self):
        return self.subsample_percentage is not None

    def is_only_targets(self):
        return self.mode == "only_targets"

    def is_default(self):
        return self.mode.startswith("default")

    def is_sequential(self):
        return self.mode.startswith("sequential")

    def is_generation(self):
        return self.mode.endswith("generation")

    def is_classification(self):
        return self.mode.endswith("classification")

    def attach_name_to(self, name):
        if self.is_default():
            name += "-def"
        if self.is_sequential():
            name += "-seq"
        if self.is_classification():
            name += "-cls"
        if self.is_generation():
            name += "-gen"
        if self.is_subsample():
            if isinstance(self.subsample_percentage, int):
                name += f"-p{self.subsample_percentage}"
            else:
                name += f"-p{self.subsample_percentage:.1f}"
        return name

    def __str__(self):
        return self.mode

    def __repr__(self):
        return f"DataMode({self.mode})"


class PentoBoardsDataset(torch.utils.data.Dataset):
    """ The dataset we use for the experiments

    Annotations:
        [
         {'id': 133,
         'size': 5,
         'pieces': [['cyan', 'P', 'bottom center', 0],
              ['green', 'U', 'bottom center', 0],
              ['navy blue', 'X', 'right center', 90],
              ['green', 'V', 'top center', 0],
              ['navy blue', 'V', 'left center', 0]],
         'target': 0,
         'refs': [{'user': 'ia',
           'instr': 'Take the cyan piece',
           'type': 0,
           'props': {'color': 'cyan'}}],
         'bboxes': [[119, 134, 186, 209],
              [104, 119, 171, 194],
              [186, 209, 82, 104],
              [119, 141, 7, 29],
              [29, 52, 104, 126]],
         'global_id': 2653}
        ]
    Note: Bounding Boxes are (x_min, x_max, y_min, y_max)

    Images are numpy arrays in the hdf5 file at the anno_id index.
    """

    def __init__(self, vocab, category_name: str, split_name: str, data_dir: str, mode: DataMode,
                 max_pieces: int = None):
        self.mode = mode if isinstance(mode, DataMode) else DataMode(mode)
        if self.mode.is_sequential():
            if max_pieces is None:
                raise Exception("'max_pieces' must be given when mode is", self.mode)
            self.max_pieces = max_pieces
        self.vocab = vocab
        if self.mode.is_generation():
            self.tokenizer = get_tokenizer()
            self.end_token = vocab(["<e>"])  # here we use the returned list
            self.pad_token = vocab(["<p>"])[0]
        self.category = category_name
        self.split = split_name
        self.data_name = f"{category_name}_{split_name}"
        self.annotations = load_annotations(data_dir, self.data_name)
        if category_name == "data":
            print("Category 'data' detected")
            file_path = os.path.join(data_dir, "data.boards.hdf5")
        else:
            file_path = os.path.join(data_dir, self.data_name + ".boards.hdf5")
        self.images_files = h5py.File(file_path, "r")  # close later again!
        print(f"Loaded {len(self.images_files['images'])} images")
        self.height, self.width, self.channels = self.images_files["images"][0].shape  # probe for sizes
        self.piece_transform = transforms.Compose([ToTensor(),
                                                   Resize(size=(self.height, self.width)),
                                                   RandomAffine(degrees=0, translate=(.05, .05), fill=1.)])
        self.context_transform = transforms.Compose([ToTensor()])
        if self.mode.is_sequential():
            self.pad_image = torch.zeros((self.channels, self.height, self.width))
            self.pad_attribute = torch.zeros(5)
        # "flatten" annotations based on refs.
        # For now there is only one ref per annotation, but there might be more in future
        self.refs = [(ref, anno) for anno in self.annotations for ref in anno["refs"]]
        if mode.is_subsample() and split_name == "train":
            total = len(self.refs)
            subsample_size = int(total / 100 * mode.subsample_percentage)
            print(f"Sub-sample {subsample_size} data points (p={mode.subsample_percentage})")
            self.refs = random.sample(self.refs, k=subsample_size)

    def close(self):
        self.images_files.close()

    def __get_inputs(self, index):
        _, anno = self.refs[index]
        global_id = anno["global_id"]
        context_image = self.images_files["images"][anno["group_id"]]
        target_idx, bboxes = anno["target"], anno["bboxes"]
        target_image, target_attributes = crop_piece(context_image, target_idx, bboxes)
        target_image = self.piece_transform(target_image)
        if self.mode.is_sequential():
            context_images = []
            context_attributes = []
            #  If a ByteTensor is provided, the non-zero positions are not allowed to attend
            #  while the zero positions will be unchanged.
            context_padding = []
            for piece_idx in range(len(anno["pieces"])):
                if piece_idx != target_idx:
                    piece_image, piece_attributes = crop_piece(context_image, piece_idx, bboxes)
                    context_images.append(self.piece_transform(piece_image))
                    context_attributes.append(piece_attributes)
                    context_padding.append(0)
            # perform "padding" for the image sequence
            # substract one for the target piece (to be prepended)
            while len(context_images) < self.max_pieces - 1:
                context_images.append(self.pad_image)
                context_attributes.append(self.pad_attribute)
                context_padding.append(1)
            context_images = torch.stack(context_images, dim=0)
            context_attributes = torch.stack(context_attributes, dim=0)
            return target_image, context_images, target_attributes, context_attributes, \
                torch.tensor(context_padding), torch.tensor(global_id)
        else:  # default
            context_image = self.context_transform(context_image)
            return target_image, context_image, target_attributes, torch.tensor(global_id)

    def __get_labels(self, index):
        ref, anno = self.refs[index]
        encoding = None

        if self.mode.is_only_targets():
            target_idx = anno["target"]
            target = anno["pieces"][target_idx]
            instr = "Take the {} {} in the {}".format(target[0], target[1], target[2])
            tokens = self.tokenizer(instr)
            # Note: no start token here, this is automatically given by the model internals
            encoding = self.vocab(tokens) + self.end_token

        if self.mode.is_generation():
            tokens = self.tokenizer(ref["instr"])
            # Note: no start token here, this is automatically given by the model internals
            encoding = self.vocab(tokens) + self.end_token

        if self.mode.is_classification():
            encoding = ref["sent_type"]

        if encoding is None:
            raise Exception("No labels found for mode " + str(self.mode))

        return torch.tensor(encoding)

    def __getitem__(self, index):
        inputs = self.__get_inputs(index)
        labels = self.__get_labels(index)
        return inputs, labels

    def collate(self, data):
        inputs = [d[0] for d in data]
        inputs = default_collate(inputs)
        if self.mode.is_sequential():
            inputs = {
                "target_image": inputs[0],
                "context_images": inputs[1],
                "target_attributes": inputs[2],
                "context_attributes": inputs[3],
                "context_paddings": inputs[4],
                "global_id": inputs[5],
            }
        else:  # default
            inputs = {
                "target_image": inputs[0],
                "context_images": inputs[1],
                "target_attributes": inputs[2],
                "global_id": inputs[3],
            }

        labels = [d[1] for d in data]
        if self.mode.is_generation():  # varying length sequence
            labels = pad_sequence(sequences=labels, padding_value=self.pad_token, batch_first=True)
        else:  # classification
            labels = default_collate(labels)
        return inputs, labels

    def set_data(self, data):
        self.refs = data

    def get_data(self):
        return self.refs

    def __len__(self):
        return len(self.refs)
