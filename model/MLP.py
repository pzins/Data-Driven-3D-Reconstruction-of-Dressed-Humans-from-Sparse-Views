import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, filter_channels, num_views, view_fusion, grid_res, grid_dim, partial_grid=False, no_residual=True, activation=nn.LeakyReLU(), last_op=nn.Sigmoid(), intermediate_result=False):
        super(MLP, self).__init__()
        self.grid_res = grid_res
        self.grid_dim = grid_dim
        self.partial_grid = partial_grid
        self.filters =  nn.ModuleList()
        self.filter_channels = filter_channels
        self.activation = activation
        self.last_op = last_op
        self.view_fusion = view_fusion
        self.no_residual = no_residual
        self.intermediate_result = intermediate_result
        
        if self.view_fusion == "attention":
            self.num_views = 1
        else:
            self.num_views = num_views
        
        if self.view_fusion == "concat":
            self.filter_channels[0] *= self.num_views


        for l in range(0, len(self.filter_channels) - 1):
            if self.no_residual:
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            self.filter_channels[l],
                            self.filter_channels[l + 1],
                            1))
                else:
                    if self.partial_grid:
                        self.filters.append(nn.Conv1d(
                            self.filter_channels[l],
                            self.filter_channels[l + 1],
                            self.grid_res*self.grid_dim - (self.grid_dim - 1),
                            self.grid_res*self.grid_dim - (self.grid_dim - 1)))
                    else:
                        self.filters.append(nn.Conv1d(
                            self.filter_channels[l],
                            self.filter_channels[l + 1],
                            self.grid_res**self.grid_dim,
                            self.grid_res**self.grid_dim))
            else:
                if 0 != l:
                        self.filters.append(
                        nn.Conv1d(
                            self.filter_channels[l] + self.filter_channels[0],
                            self.filter_channels[l + 1],
                            1))
            
                else:
                    if self.partial_grid:
                        self.filters.append(nn.Conv1d(
                            self.filter_channels[l],
                            self.filter_channels[l + 1],
                            self.grid_res*self.grid_dim - (self.grid_dim - 1),
                            self.grid_res*self.grid_dim - (self.grid_dim - 1)))
                    else:
                        self.filters.append(nn.Conv1d(
                            self.filter_channels[l],
                            self.filter_channels[l + 1],
                            self.grid_res**self.grid_dim,
                            self.grid_res**self.grid_dim))

            # self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        y = feature
        
        if self.view_fusion =="concat":
            y = y.view(y.shape[0] // self.num_views, y.shape[1] * self.num_views, -1)
        
        if self.partial_grid:
            idx = np.arange(
                    (self.grid_res*self.grid_dim - (self.grid_dim - 1)) // 2,
                    y.shape[2],
                    self.grid_res*self.grid_dim - (self.grid_dim - 1)
            )
        else:
            idx = np.arange(
                self.grid_res**self.grid_dim // 2,
                y.shape[2],
                self.grid_res**self.grid_dim
            )
        
        geo_feat = None
        tmpy = y[:, :, idx]
        for i, f in enumerate(self.filters):
            if self.no_residual:
                y = f(y)
            else:
                y = f(
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            if i != len(self.filters) - 1 and self.activation:
                y = self.activation(y)
            
            if i == len(self.filters) // 2:
                geo_feat = y
            if self.view_fusion == "mean" and self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)
                tmpy = tmpy[:, :, idx]

        if self.last_op:
            y = self.last_op(y)
        if self.intermediate_result:
            return y, geo_feat
        return y
