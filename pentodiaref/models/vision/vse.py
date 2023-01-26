import torch
from torch import nn

from pentodiaref.models.vision.resnet import LitPieceNet


class VisualSequenceEncoding(nn.Module):
    """ Embed images and add region+type information """

    def __init__(self, d_model: int, layer_norm_eps: float, device):
        super().__init__()
        self.device = device
        self.ablation_mode = None  # might be set externally

        self.piece_net = LitPieceNet()

        self.feature_projection = nn.Sequential(nn.Linear(self.piece_net.image_embedding_dims, d_model),
                                                nn.LayerNorm(d_model, eps=layer_norm_eps))
        self.attribute_projection = nn.Sequential(nn.Linear(5, d_model),
                                                  nn.LayerNorm(d_model, eps=layer_norm_eps))
        # 0: target, 1: padding, 2: distractor
        self.type_embeddings = nn.Sequential(nn.Embedding(3, d_model, padding_idx=1),
                                             nn.LayerNorm(d_model, eps=layer_norm_eps))

    def forward(self, inputs):
        target_images = inputs["target_image"]
        context_images = inputs["context_images"]
        target_attributes = inputs["target_attributes"]

        # B x L x C x H x W
        piece_images = torch.cat([target_images.unsqueeze(dim=1), context_images], dim=1)  # pre-pend the targets
        # B x L x E
        piece_features = self.piece_net.forward_seq(piece_images)
        if self.ablation_mode == "replace_random":
            piece_features = torch.randn_like(piece_features)
        if self.ablation_mode == "add_random":
            piece_features = piece_features + torch.randn_like(piece_features)
        proj_features = self.feature_projection(piece_features)
        batch_size = piece_images.shape[0]

        context_attributes = inputs["context_attributes"]
        piece_attributes = torch.cat([target_attributes.unsqueeze(dim=1), context_attributes], dim=1)
        proj_attributes = self.attribute_projection(piece_attributes)

        if self.ablation_mode == "random_regions":
            proj_attributes = torch.randn_like(proj_attributes)

        # 0: target, 1: padding, 2: distractor
        context_paddings = inputs["context_paddings"]
        type_tokens = context_paddings.clone()  # true padding is already at 1
        type_tokens[context_paddings == 0] = 2  # set everything else to distractor (which is not padding)
        target_tokens = torch.full((batch_size, 1), fill_value=0, device=context_paddings.device)
        type_tokens = torch.cat([target_tokens, type_tokens], dim=1)  # prepend target as 0
        type_embedding = self.type_embeddings(type_tokens)

        if self.ablation_mode == "random_types":
            type_embedding = torch.randn_like(type_embedding)

        features = (proj_features + proj_attributes + type_embedding) / 3
        padding_with_targets = torch.cat([target_tokens, context_paddings], dim=1)  # add targets to pad_mask

        features = torch.permute(features, dims=[1, 0, 2])  # seq_len first
        return features, padding_with_targets
