"""Per-query Gaussian hierarchy, independent of image sampling."""
import torch
from torch import nn


class GaussianHierarchy(nn.Module):
    def __init__(self, dim, parents, children, heads=8,
                 use_node_hidden=True, sibling_attention_mode='sibling_only',
                 group_readout_mode='cross_attention', dropout=0.1):
        super().__init__()
        if parents < 1 or children < parents:
            raise ValueError('Each parent must have at least one child')
        if sibling_attention_mode not in ('none', 'sibling_only'):
            raise ValueError('Only sibling-only attention is supported')
        if group_readout_mode not in ('none', 'mean_pool', 'cross_attention'):
            raise ValueError('Unknown group readout mode')
        self.parents, self.num_children = parents, children
        self.use_node_hidden = use_node_hidden
        self.sibling_attention_mode = sibling_attention_mode
        self.group_readout_mode = group_readout_mode
        # Consecutive children belong to the same parent; distribute remainders
        # to the first parents so non-integral branching remains deterministic.
        counts = [children // parents + (i < children % parents) for i in range(parents)]
        parent_index = torch.repeat_interleave(torch.arange(parents), torch.tensor(counts))
        slots = torch.cat([torch.arange(n) for n in counts])
        self.register_buffer('parent_index', parent_index, persistent=False)
        self.register_buffer('child_slot', slots, persistent=False)
        self.counts = counts
        self.parent_proj = nn.Linear(dim, dim) if use_node_hidden else None
        self.group_proj = nn.Linear(dim, dim)
        self.child_embed = nn.Embedding(max(counts), dim)
        self.node_norm = nn.LayerNorm(dim)
        if sibling_attention_mode == 'sibling_only' and max(counts) > 1:
            self.sibling_norm = nn.LayerNorm(dim)
            self.sibling_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.dropout = nn.Dropout(dropout)
        if group_readout_mode == 'cross_attention':
            self.group_read_norm = nn.LayerNorm(dim)
            self.node_read_norm = nn.LayerNorm(dim)
            self.group_read_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        if group_readout_mode != 'none':
            self.group_read_proj = nn.Linear(dim, dim)

    def forward(self, group, parent):
        b, q, d = group.shape
        if parent.shape != (b, q, self.parents, d):
            raise ValueError('Parent hidden state shape does not match hierarchy')
        parent_features = self.parent_proj(parent[:, :, self.parent_index]) if self.use_node_hidden else 0
        nodes = self.node_norm(parent_features + self.group_proj(group).unsqueeze(2)
                               + self.child_embed(self.child_slot))
        if self.sibling_attention_mode == 'sibling_only' and max(self.counts) > 1:
            # Batch parents of the same size, without padding or cross-parent attention.
            parts = []
            start = 0
            parent_start = 0
            while parent_start < self.parents:
                count = self.counts[parent_start]
                parent_end = parent_start + 1
                while parent_end < self.parents and self.counts[parent_end] == count:
                    parent_end += 1
                nparents = parent_end - parent_start
                end = start + nparents * count
                x = nodes[:, :, start:end].reshape(b * q * nparents, count, d)
                if count > 1:
                    norm = self.sibling_norm(x)
                    x = x + self.dropout(self.sibling_attn(norm, norm, norm, need_weights=False)[0])
                parts.append(x.reshape(b, q, nparents * count, d))
                start = end
                parent_start = parent_end
            nodes = torch.cat(parts, dim=2)
        if self.group_readout_mode == 'cross_attention':
            query = self.group_read_norm(group).reshape(b * q, 1, d)
            kv = self.node_read_norm(nodes).reshape(b * q, self.num_children, d)
            readout = self.group_read_attn(query, kv, kv, need_weights=False)[0]
            group = group + self.group_read_proj(readout.reshape(b, q, d))
        elif self.group_readout_mode == 'mean_pool':
            group = group + self.group_read_proj(nodes.mean(dim=2))
        return group, nodes


def slot_forward(branch, nodes, output_dim):
    """Apply shared hidden layers and each slot's original output rows."""
    hidden = branch[:-1](nodes)
    output = branch[-1]
    weights = output.weight.reshape(nodes.shape[2], output_dim, nodes.shape[-1])
    bias = output.bias.reshape(nodes.shape[2], output_dim)
    return torch.einsum('bqrd,rod->bqro', hidden, weights) + bias
