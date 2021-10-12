# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from typing import Tuple
from torch import nn, Tensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: Tensor):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_positions=50, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(num_positions, num_pos_feats)
        self.col_embed = nn.Embedding(num_positions, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: Tensor):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class BoxEmbeddingLearned(nn.Module):
    """
    Absolute box embedding, learned.
    """
    def __init__(self, num_positions=50, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(num_positions, num_pos_feats)
        self.col_embed = nn.Embedding(num_positions, num_pos_feats)
        self.hei_embed = nn.Embedding(num_positions, num_pos_feats)
        self.wid_embed = nn.Embedding(num_positions, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.hei_embed.weight)
        nn.init.uniform_(self.wid_embed.weight)

    def forward(self, scale: Tuple, boxes: Tensor):
        scale_x, scale_y = scale[0], scale[1]
        boxes[:, 0::2] = boxes[:, 0::2] * scale_x
        boxes[:, 1::2] = boxes[:, 1::2] * scale_y
        boxes = boxes.long()
        x_emb = self.col_embed(boxes[:, 0])
        y_emb = self.row_embed(boxes[:, 1])
        w_emb = self.wid_embed(boxes[:, 2])
        h_emb = self.hei_embed(boxes[:, 3])
        return torch.cat([x_emb, y_emb, w_emb, h_emb], dim=-1)
        
        
class BoxEmbeddingSine(nn.Module):
    """
    Module for generating the position embeddings for the detected boxes.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True,
                 scale=None, encode_shape=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.encode_shape = encode_shape
    
    def forward(self, image_size: Tuple, boxes: Tensor):
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=boxes.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        boxes = box_clip(image_size, boxes)

        scale_x, scale_y = (1. / image_size[1] , 1. / image_size[0])
        boxes[:, 0::2] *= scale_x
        boxes[:, 1::2] *= scale_y
        
        xc = (boxes[:, 0] + boxes[:, 2]) / 2
        yc = (boxes[:, 1] + boxes[:, 3]) / 2
        w  = (boxes[:, 2] - boxes[:, 0])
        h  = (boxes[:, 3] - boxes[:, 1])
        
        x_embed = xc * self.scale
        y_embed = yc * self.scale
        w_embed = w  * self.scale
        h_embed = h  * self.scale

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_w = w_embed[:, None] / dim_t
        pos_h = h_embed[:, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_w = torch.stack((pos_w[:, 0::2].sin(), pos_w[:, 1::2].cos()), dim=2).flatten(1)
        pos_h = torch.stack((pos_h[:, 0::2].sin(), pos_h[:, 1::2].cos()), dim=2).flatten(1)
        
        if self.encode_shape:
            return torch.cat((pos_x, pos_y, pos_w, pos_h), dim=-1)
        else:
            return torch.cat((pos_x, pos_y), dim=-1)


def box_clip(image_size: Tuple, boxes: Tensor):
    """
    Clip (in place) the boxes by limiting x coordinates to the range [0, width]
    and y coordinates to the range [0, height].

    Args:
        box_size (height, width): The clipping box's size.
    """
    assert torch.isfinite(boxes).all(), "Box tensor contains infinite or NaN!"
    h, w = image_size
    boxes[:, 0].clamp_(min=0, max=w)
    boxes[:, 1].clamp_(min=0, max=h)
    boxes[:, 2].clamp_(min=0, max=w)
    boxes[:, 3].clamp_(min=0, max=h)
    return boxes


def build_pos_encoding(hidden_dim: int = 256, position_embedding_type: str = "sine"):
    """
    Build the position encoder for all positions in the feature map.
    Args:
        hidden_dim (int): the dimension of the positional embedding.
        position_embedding_type (str): the type of positional embedding ("learned" or "sine").
    """
    N_steps = hidden_dim // 2
    if position_embedding_type in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding_type in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding_type}")

    return position_embedding


def build_box_encoding(hidden_dim: int = 256, position_embedding_type: str = "sine"):
    """
    Build the box encoder for the query.
    Args:
        hidden_dim (int): the dimension of the box embedding.
        encoder_shape (bool): if true, encode the width and height (w, h) of the query box.
            Otherwise, only the coordinate of the center point (x, y) is encoded.
    """
    N_steps = hidden_dim // 4
    if position_embedding_type == "sine":
        box_embedding = BoxEmbeddingSine(N_steps, normalize=True)
    elif position_embedding_type == "learned":
        box_embedding = BoxEmbeddingLearned(N_steps)
    return box_embedding