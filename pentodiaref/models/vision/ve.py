import torch
from torch import nn

from pentodiaref.models.vision.resnet import LitPieceNet


class VisualEncoding(nn.Module):
    """ Embed images and add region+type information """

    def __init__(self, d_model: int, layer_norm_eps: float, device):
        super().__init__()
        self.device = device
        self.ablation_mode = None  # might be set externally
        self.piecenet = LitPieceNet()
        self.feature_projection = nn.Sequential(nn.Linear(self.piecenet.image_embedding_dims, d_model),
                                                nn.LayerNorm(d_model, eps=layer_norm_eps))
        self.attribute_projection = nn.Sequential(nn.Linear(5, d_model),
                                                  nn.LayerNorm(d_model, eps=layer_norm_eps))

    def forward(self, inputs):
        target_images = inputs["target_image"]
        context_images = inputs["context_images"]
        target_attributes = inputs["target_attributes"]
        # there is only a single context image
        context_images = context_images.unsqueeze(dim=1)
        # B x L x C x H x W
        piece_images = torch.cat([target_images.unsqueeze(dim=1), context_images], dim=1)  # pre-pend the targets
        # B x L x E
        piece_features = self.piecenet.forward_seq(piece_images)
        if self.ablation_mode == "replace_random":
            piece_features = torch.randn_like(piece_features)
        if self.ablation_mode == "add_random":
            piece_features = piece_features + torch.randn_like(piece_features)
        proj_features = self.feature_projection(piece_features)
        batch_size = piece_images.shape[0]
        # no type embeddings, but simply [attributes, target, context]
        # prepend target attributes, should result into s.t. like B x 3 x 512
        proj_attributes = self.attribute_projection(target_attributes)
        features = torch.cat([proj_attributes.unsqueeze(dim=1), proj_features], dim=1)
        # no padding (all zeros), attend all
        padding_with_targets = torch.zeros((batch_size, 3), device=proj_attributes.device)
        features = torch.permute(features, dims=[1, 0, 2])  # seq_len first
        return features, padding_with_targets
