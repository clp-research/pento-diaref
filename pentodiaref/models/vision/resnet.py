from typing import Any, List
import torchvision.models.resnet
from torch import nn
import pytorch_lightning as pl

# workaround to store embedding projector files
# https://github.com/pytorch/pytorch/issues/30966
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class LitPieceNet(pl.LightningModule):
    """
        PieceNet is a wrapper around a ResNet that might be pre-trained for the prediction of piece attributes.
        We found no major improvement when pre-training on pieces for the down-stream model, so that we only
        fine-tune the ResNet during the experiments. We removed the fine-tuning code for a better overview.
    """

    # noinspection PyUnusedLocal
    def __init__(self):
        super(LitPieceNet, self).__init__()
        self.resnet = torchvision.models.resnet.resnet34(pretrained=True, progress=False)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.image_embedding_dims = 512

        self.external_logger = None
        self.external_step = 0

    def freeze_parameters(self):
        print("Freeze PieceNet parameters")
        parameters = list(self.parameters())
        for parameter in parameters:  # freeze pre-trained target piecenet
            parameter.requires_grad = False

    def finetune_last_layer(self):
        print("Fine-tune last PieceNet layer")
        parameters = list(self.parameters())
        parameters[-1].requires_grad = True

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitPieceNet")
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--arch", type=str, default="resnet34", help="[resnet34, resnet101]")
        return parent_parser

    def forward_seq(self, piece_images):
        """ Apply this model on a sequence of images (without attributes) """
        # explode the batch (B x L x H x W x C) -> (B*L x H x W x C)
        batch_size, lengths, height, width, channels = piece_images.shape
        piece_images = piece_images.reshape(-1, height, width, channels)
        piece_features = self.forward(piece_images)
        # collect as batches over sequences again
        piece_features = piece_features.reshape(batch_size, lengths, self.image_embedding_dims)
        return piece_features

    def forward(self, x) -> Any:
        image = x
        embedding = self.resnet(image).squeeze()
        if self.external_logger is not None:
            self.external_logger.experiment.add_embedding(embedding,
                                                          label_img=image,
                                                          global_step=self.external_step)
            self.external_step += 1
        return embedding
