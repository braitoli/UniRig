import torch
import torch.nn as nn
import math

class PureSparseConvTensor:
    def __init__(self, features, indices, spatial_shape=None, batch_size=None):
        self.features = features
        self.indices = indices  # (N, 4): [b, z, y, x]
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    def replace_feature(self, new_features):
        return PureSparseConvTensor(new_features, self.indices, self.spatial_shape, self.batch_size)

class PureSubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, algo=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # spconv weight shape is (out_channels, kD, kH, kW, in_channels)
        self.weight = nn.Parameter(torch.empty(out_channels, *kernel_size, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: PureSparseConvTensor) -> PureSparseConvTensor:
        feats = x.features  # (N, Cin)
        indices = x.indices.long()  # (N, 4): [b, z, y, x]
        N, Cin = feats.shape
        device = feats.device

        min_coords = indices.min(dim=0).values
        shifted = indices - min_coords
        max_coords = shifted.max(dim=0).values + 1
        
        stride_x = 1
        stride_y = max_coords[3].item() + 1
        stride_z = stride_y * (max_coords[2].item() + 1)
        stride_b = stride_z * (max_coords[1].item() + 1)
        
        hash_id = (shifted[:, 0] * stride_b +
                   shifted[:, 1] * stride_z +
                   shifted[:, 2] * stride_y +
                   shifted[:, 3] * stride_x)

        sort_perm = torch.argsort(hash_id)
        sorted_hash = hash_id[sort_perm]

        kD, kH, kW = self.kernel_size
        pD, pH, pW = kD // 2, kH // 2, kW // 2

        out_feats = torch.zeros(N, self.out_channels, device=device, dtype=feats.dtype)
        W_flat = self.weight.permute(1, 2, 3, 4, 0).reshape(-1, Cin, self.out_channels)

        k_idx = 0
        for dz in range(-pD, pD + 1):
            for dy in range(-pH, pH + 1):
                for dx in range(-pW, pW + 1):
                    target_b = shifted[:, 0]
                    target_z = shifted[:, 1] + dz
                    target_y = shifted[:, 2] + dy
                    target_x = shifted[:, 3] + dx

                    valid_coord = ((target_z >= 0) & (target_z < max_coords[1]) &
                                   (target_y >= 0) & (target_y < max_coords[2]) &
                                   (target_x >= 0) & (target_x < max_coords[3]))

                    target_hash = (target_b * stride_b +
                                   target_z * stride_z +
                                   target_y * stride_y +
                                   target_x * stride_x)

                    pos = torch.searchsorted(sorted_hash, target_hash)
                    pos_clamped = pos.clamp(max=N - 1)
                    
                    matched = valid_coord & (pos < N) & (sorted_hash[pos_clamped] == target_hash)
                    
                    if matched.any():
                        src_indices = sort_perm[pos_clamped[matched]]
                        dst_indices = torch.where(matched)[0]
                        
                        w_k = W_flat[k_idx]
                        contrib = torch.matmul(feats[src_indices], w_k)
                        out_feats.index_add_(0, dst_indices, contrib)

                    k_idx += 1

        if self.bias is not None:
            out_feats += self.bias

        return x.replace_feature(out_feats)

class _ModulesProxy:
    @staticmethod
    def is_spconv_module(module):
        return isinstance(module, PureSubMConv3d)

class SpconvFallback:
    SubMConv3d = PureSubMConv3d
    SparseConvTensor = PureSparseConvTensor
    modules = _ModulesProxy
