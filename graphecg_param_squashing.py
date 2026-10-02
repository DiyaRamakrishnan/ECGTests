"""
Shrinks GraphECG to Conv-RGNN's exact size (60,550 params) to see if GraphECG
wins on PTB-XL 6-class because of its architecture or just because it's ~20x bigger.

Result: the squashed GraphECG (60,543 params) ties Conv-RGNN, so most of the
full model's advantage comes from having more parameters.

Run:
  python graphecg_param_squashing.py search --target_total 60550
  python graphecg_param_squashing.py breakdown
  python graphecg_param_squashing.py train --model graphecg_squashed_v2 --seed 42 --epochs 50
  python graphecg_param_squashing.py aggregate --results_dir $SAVE_LOCATION/graphecg_squash_ptbxl6class

Needs DATASET_LOCATION (PTB-XL folder) and SAVE_LOCATION (output folder).
"""
import os
import ast
import json
import math
import argparse
import random
import itertools

import numpy as np
import pandas as pd
import scipy.signal as sig
import wfdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch

from sklearn.metrics import roc_auc_score, f1_score


LABEL_NAMES = ["NORM", "STTC", "CD", "HYP", "ASMI", "IMI"]

# full GraphECG's param split -- the squashed version keeps the same ratios
FULL_TOTAL = 1_207_430
FULL_ENCODER = 240_320
FULL_GNN = 937_344
ENC_FRAC_TARGET = FULL_ENCODER / FULL_TOTAL
GNN_FRAC_TARGET = FULL_GNN / FULL_TOTAL


ELECTRODE_POSITIONS = {
    'WCT': np.array([0.0, 0.0, 0.0], dtype=np.float32),
    'RA': np.array([-0.449, -0.596, 0.174], dtype=np.float32),
    'LA': np.array([0.551, -0.096, 0.174], dtype=np.float32),
    'LL': np.array([-0.101, 0.693, 0.674], dtype=np.float32),
    'V1': np.array([0.999, -0.017, 0.0], dtype=np.float32),
    'V2': np.array([0.996, 0.087, 0.0], dtype=np.float32),
    'V3': np.array([0.949, 0.259, 0.174], dtype=np.float32),
    'V4': np.array([0.813, 0.470, 0.342], dtype=np.float32),
    'V5': np.array([0.663, 0.663, 0.342], dtype=np.float32),
    'V6': np.array([0.500, 0.866, 0.0], dtype=np.float32),
    'mid_LA_LL': np.array([0.225, 0.298, 0.424], dtype=np.float32),
    'mid_RA_LL': np.array([-0.275, 0.048, 0.424], dtype=np.float32),
    'mid_RA_LA': np.array([0.051, -0.346, 0.174], dtype=np.float32),
}
ALL_ELECTRODES = list(ELECTRODE_POSITIONS.keys())

LEAD_DEFINITIONS = {
    'I': ('RA', 'LA'), 'II': ('RA', 'LL'), 'III': ('LA', 'LL'),
    'aVR': ('mid_LA_LL', 'RA'), 'aVL': ('mid_RA_LL', 'LA'), 'aVF': ('mid_RA_LA', 'LL'),
    'V1': ('WCT', 'V1'), 'V2': ('WCT', 'V2'), 'V3': ('WCT', 'V3'),
    'V4': ('WCT', 'V4'), 'V5': ('WCT', 'V5'), 'V6': ('WCT', 'V6'),
}
GRAPHECG_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def _direction_to_spherical(src_pos, tgt_pos):
    direction = tgt_pos - src_pos
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return 0.0, 0.0
    d = direction / norm
    phi = math.acos(np.clip(d[2], -1.0, 1.0))
    theta = math.atan2(d[1], d[0])
    if theta < 0:
        theta += 2 * math.pi
    return theta, phi


class ECGGraphBuilder:
    def __init__(self):
        self.electrodes = ALL_ELECTRODES
        self.electrode_to_idx = {}
        for i, e in enumerate(self.electrodes):
            self.electrode_to_idx[e] = i
        position_list = [ELECTRODE_POSITIONS[e] for e in self.electrodes]
        self.positions = torch.tensor(np.array(position_list), dtype=torch.float32)

    def build_graph(self, signals, bidirectional=True):
        edge_src = []
        edge_tgt = []
        edge_attr_list = []
        edge_spherical = []
        for lead_name, signal in signals.items():
            if lead_name not in LEAD_DEFINITIONS:
                raise ValueError(f"Unknown lead: {lead_name}")
            src, tgt = LEAD_DEFINITIONS[lead_name]
            src_idx = self.electrode_to_idx[src]
            tgt_idx = self.electrode_to_idx[tgt]
            signal = np.asarray(signal, dtype=np.float32)

            edge_src.append(src_idx)
            edge_tgt.append(tgt_idx)
            edge_attr_list.append(signal)
            theta, phi = _direction_to_spherical(ELECTRODE_POSITIONS[src], ELECTRODE_POSITIONS[tgt])
            edge_spherical.append([theta, phi])

            if bidirectional:
                # reverse edge gets the flipped signal
                edge_src.append(tgt_idx)
                edge_tgt.append(src_idx)
                edge_attr_list.append(-signal)
                theta, phi = _direction_to_spherical(ELECTRODE_POSITIONS[tgt], ELECTRODE_POSITIONS[src])
                edge_spherical.append([theta, phi])

        return Data(
            x=self.positions.clone(),
            edge_index=torch.tensor([edge_src, edge_tgt], dtype=torch.long),
            edge_attr=torch.tensor(np.stack(edge_attr_list), dtype=torch.float32),
            edge_spherical=torch.tensor(edge_spherical, dtype=torch.float32),
        )

    def build_from_array(self, ecg, lead_indices=None, bidirectional=True):
        if ecg.shape[0] != 12:
            ecg = ecg.T
        if lead_indices is None:
            lead_indices = list(range(12))
        signals = {}
        for i in lead_indices:
            signals[GRAPHECG_LEAD_ORDER[i]] = ecg[i]
        return self.build_graph(signals, bidirectional=bidirectional)


class ResBlock1d(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.Mish(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class SphericalHarmonicEncoding(nn.Module):
    def __init__(self, max_degree=4):
        super().__init__()
        self.max_degree = max_degree
        total = 0
        for l in range(max_degree + 1):
            total += 2 * l + 1
        self.output_dim = total + 4

    def forward(self, pos):
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2).clamp(min=1e-8)
        x_n = x / r
        y_n = y / r
        z_n = z / r

        features = [torch.ones_like(x_n) * 0.5 * math.sqrt(1 / math.pi)]

        if self.max_degree >= 1:
            c = 0.5 * math.sqrt(3 / math.pi)
            features.extend([c * y_n, c * z_n, c * x_n])

        if self.max_degree >= 2:
            c2 = 0.5 * math.sqrt(15 / math.pi)
            c2_0 = 0.25 * math.sqrt(5 / math.pi)
            features.extend([c2 * x_n * y_n, c2 * y_n * z_n, c2_0 * (3 * z_n**2 - 1),
                             c2 * x_n * z_n, 0.5 * c2 * (x_n**2 - y_n**2)])

        if self.max_degree >= 3:
            features.extend([y_n * (3 * x_n**2 - y_n**2), x_n * y_n * z_n,
                             y_n * (5 * z_n**2 - 1), z_n * (5 * z_n**2 - 3),
                             x_n * (5 * z_n**2 - 1), z_n * (x_n**2 - y_n**2),
                             x_n * (x_n**2 - 3 * y_n**2)])

        if self.max_degree >= 4:
            xy = x_n * y_n
            xz = x_n * z_n
            yz = y_n * z_n
            x2 = x_n**2
            y2 = y_n**2
            z2 = z_n**2
            features.extend([xy * (x2 - y2), yz * (3 * x2 - y2), xy * (7 * z2 - 1),
                             yz * (7 * z2 - 3), 35 * z2**2 - 30 * z2 + 3,
                             xz * (7 * z2 - 3), (x2 - y2) * (7 * z2 - 1),
                             xz * (x2 - 3 * y2), x2**2 - 6 * x2 * y2 + y2**2])

        sh = torch.stack(features, dim=-1)
        radial = torch.stack([r, torch.sin(math.pi * r), torch.cos(math.pi * r), torch.exp(-r)], dim=-1)
        return torch.cat([sh, radial], dim=-1)


class SignalEncoder(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=192):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.Mish(inplace=True),
            ResBlock1d(64),
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim), nn.Mish(inplace=True),
            ResBlock1d(hidden_dim),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim), nn.Mish(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.encoder(x)
        out = self.pool(out)
        return out.squeeze(-1)


class SignalEncoderCustom(nn.Module):
    def __init__(self, stem_channels=64, hidden_dim=128, output_dim=192):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, stem_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(stem_channels), nn.Mish(inplace=True),
            ResBlock1d(stem_channels),
            nn.Conv1d(stem_channels, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim), nn.Mish(inplace=True),
            ResBlock1d(hidden_dim),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim), nn.Mish(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.encoder(x)
        out = self.pool(out)
        return out.squeeze(-1)


class ECGMessagePassing(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim), nn.Mish(inplace=True),
            nn.Linear(hidden_dim, hidden_dim))
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim), nn.Mish(inplace=True),
            nn.Linear(hidden_dim, node_dim))
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x + out)

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class GraphECG(nn.Module):
    def __init__(self, node_dim=128, edge_dim=192, hidden_dim=192,
                 num_layers=3, tabular_dim=0, num_classes=6, dropout=0.5):
        super().__init__()
        self.tabular_dim = tabular_dim
        self.num_classes = num_classes
        self.pos_encoder = SphericalHarmonicEncoding(max_degree=4)
        self.node_proj = nn.Linear(self.pos_encoder.output_dim, node_dim)
        self.signal_encoder = SignalEncoder(hidden_dim=128, output_dim=edge_dim)
        layers = []
        for _ in range(num_layers):
            layers.append(ECGMessagePassing(node_dim, edge_dim, node_dim * 2))
        self.gnn_layers = nn.ModuleList(layers)
        self.graph_proj = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.Mish(inplace=True))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim + tabular_dim, num_classes)

    def forward(self, data, tabular=None):
        batch = getattr(data, 'batch', None)
        node_features = self.node_proj(self.pos_encoder(data.x))
        edge_embed = self.signal_encoder(data.edge_attr)

        x = node_features
        for gnn in self.gnn_layers:
            x = gnn(x, data.edge_index, edge_embed)

        if batch is None:
            graph_embed = x.mean(dim=0, keepdim=True)
        else:
            graph_embed = global_mean_pool(x, batch)
        graph_embed = self.dropout(self.graph_proj(graph_embed))

        if self.tabular_dim > 0:
            if tabular is None:
                tabular = graph_embed.new_zeros(graph_embed.shape[0], self.tabular_dim)
            fused = torch.cat([graph_embed, tabular], dim=-1)
        else:
            fused = graph_embed
        return {'logits': self.classifier(fused), 'embedding': graph_embed}


class GraphECGSquashable(nn.Module):
    def __init__(self, node_dim, edge_dim, gnn_hidden_dim, num_layers,
                 stem_channels, encoder_hidden_dim, graph_proj_hidden_dim,
                 tabular_dim=0, num_classes=6, dropout=0.5):
        super().__init__()
        self.tabular_dim = tabular_dim
        self.pos_encoder = SphericalHarmonicEncoding(max_degree=4)
        self.node_proj = nn.Linear(self.pos_encoder.output_dim, node_dim)
        self.signal_encoder = SignalEncoderCustom(
            stem_channels=stem_channels, hidden_dim=encoder_hidden_dim, output_dim=edge_dim)
        layers = []
        for _ in range(num_layers):
            layers.append(ECGMessagePassing(node_dim, edge_dim, gnn_hidden_dim))
        self.gnn_layers = nn.ModuleList(layers)
        self.graph_proj = nn.Sequential(nn.Linear(node_dim, graph_proj_hidden_dim), nn.Mish(inplace=True))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(graph_proj_hidden_dim + tabular_dim, num_classes)

    def forward(self, data, tabular=None):
        batch = getattr(data, 'batch', None)
        node_features = self.node_proj(self.pos_encoder(data.x))
        edge_embed = self.signal_encoder(data.edge_attr)

        x = node_features
        for gnn in self.gnn_layers:
            x = gnn(x, data.edge_index, edge_embed)

        if batch is None:
            graph_embed = x.mean(dim=0, keepdim=True)
        else:
            graph_embed = global_mean_pool(x, batch)
        graph_embed = self.dropout(self.graph_proj(graph_embed))

        if self.tabular_dim > 0:
            if tabular is None:
                tabular = graph_embed.new_zeros(graph_embed.shape[0], self.tabular_dim)
            fused = torch.cat([graph_embed, tabular], dim=-1)
        else:
            fused = graph_embed
        return {'logits': self.classifier(fused), 'embedding': graph_embed}


# found with search() for 60,550 params -> 60,543 total
SQUASHED_V2_CONFIG = dict(
    stem_channels=21, encoder_hidden_dim=27, edge_dim=19, node_dim=31,
    gnn_hidden_dim=59, num_layers=3, graph_proj_hidden_dim=15,
)


def build_squashed(config, tabular_dim=0, num_classes=6):
    return GraphECGSquashable(
        node_dim=config["node_dim"],
        edge_dim=config["edge_dim"],
        gnn_hidden_dim=config["gnn_hidden_dim"],
        num_layers=config["num_layers"],
        stem_channels=config["stem_channels"],
        encoder_hidden_dim=config["encoder_hidden_dim"],
        graph_proj_hidden_dim=config["graph_proj_hidden_dim"],
        tabular_dim=tabular_dim,
        num_classes=num_classes,
    )


class GraphECGWrapper(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = GraphECG(num_classes=num_classes, tabular_dim=0)
        self.builder = ECGGraphBuilder()

    def forward(self, x_ecg):
        device = x_ecg.device
        x_np = x_ecg.detach().cpu().numpy()
        data_list = []
        for b in range(x_ecg.shape[0]):
            data_list.append(self.builder.build_from_array(x_np[b], bidirectional=True))
        batch = Batch.from_data_list(data_list).to(device)
        return self.model(batch)["logits"]


class GraphECGSquashedV2Wrapper(nn.Module):
    def __init__(self, num_classes=6, config=None):
        super().__init__()
        if not config:
            config = SQUASHED_V2_CONFIG
        self.model = build_squashed(config, tabular_dim=0, num_classes=num_classes)
        self.builder = ECGGraphBuilder()

    def forward(self, x_ecg):
        device = x_ecg.device
        x_np = x_ecg.detach().cpu().numpy()
        data_list = []
        for b in range(x_ecg.shape[0]):
            data_list.append(self.builder.build_from_array(x_np[b], bidirectional=True))
        batch = Batch.from_data_list(data_list).to(device)
        return self.model(batch)["logits"]


class LeadFeatureExtractor(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        B, L, T = x.shape
        x = x.reshape(B * L, 1, T)
        feat = self.proj(self.net(x).squeeze(-1))
        return feat.reshape(B, L, -1)


class ConvRGNN(nn.Module):
    def __init__(self, feat_dim=64, hidden_dim=64, num_classes=6, num_layers=3):
        super().__init__()
        self.extractor = LeadFeatureExtractor(out_dim=feat_dim)
        self.conv_in = GCNConv(feat_dim, hidden_dim)
        res_convs = []
        for _ in range(num_layers):
            res_convs.append(GCNConv(hidden_dim, hidden_dim))
        self.res_convs = nn.ModuleList(res_convs)
        bns = []
        for _ in range(num_layers):
            bns.append(nn.BatchNorm1d(hidden_dim))
        self.bns = nn.ModuleList(bns)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def build_graph(self, x_ecg, device):
        B = x_ecg.shape[0]
        feats = self.extractor(x_ecg)
        limb = [0, 1, 2, 3, 4, 5]
        chest = [6, 7, 8, 9, 10, 11]
        bridge = [0, 5, 9, 10]
        edges = []
        for i in limb:
            for j in limb:
                if i != j:
                    edges.append((i, j))
        for i in chest:
            for j in chest:
                if i != j:
                    edges.append((i, j))
        for i in bridge:
            for j in bridge:
                if i != j and (i, j) not in edges:
                    edges.append((i, j))
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
        data_list = []
        for b in range(B):
            data_list.append(Data(x=feats[b], edge_index=edge_index))
        return Batch.from_data_list(data_list)

    def forward(self, x_ecg):
        device = x_ecg.device
        batch = self.build_graph(x_ecg, device)
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        x = F.relu(self.conv_in(x, edge_index))
        for conv, bn in zip(self.res_convs, self.bns):
            residual = x
            x = F.relu(bn(conv(x, edge_index)))
            x = x + residual
        return self.classifier(global_mean_pool(x, batch_idx))


MODEL_MAP = {
    "conv_rgnn": ConvRGNN,
    "graphecg": GraphECGWrapper,
    "graphecg_squashed_v2": GraphECGSquashedV2Wrapper,
}


def count_params(module):
    total = 0
    for p in module.parameters():
        total += p.numel()
    return total


def search_encoder(enc_target, output_dim_pin=None,
                   stem_range=range(2, 25), hidden_range=range(4, 81), output_range=range(2, 65)):
    best = None
    best_err = float("inf")
    if output_dim_pin is not None:
        output_candidates = [output_dim_pin]
    else:
        output_candidates = output_range
    for c0, h, o in itertools.product(stem_range, hidden_range, output_candidates):
        enc = SignalEncoderCustom(stem_channels=c0, hidden_dim=h, output_dim=o)
        n = count_params(enc)
        err = abs(n - enc_target)
        if err < best_err:
            best_err = err
            best = (c0, h, o, n)
    return best


def search_gnn(gnn_target, node_range=range(4, 33), edge_range=range(4, 33),
               hidden_range=range(4, 65), layer_opts=(2, 3)):
    best = None
    best_err = float("inf")
    for node_dim, edge_dim, gnn_hidden, num_layers in itertools.product(
        node_range, edge_range, hidden_range, layer_opts
    ):
        layer_list = []
        for _ in range(num_layers):
            layer_list.append(ECGMessagePassing(node_dim, edge_dim, gnn_hidden))
        layers = nn.ModuleList(layer_list)
        n = count_params(layers)
        err = abs(n - gnn_target)
        if err < best_err:
            best_err = err
            best = (node_dim, edge_dim, gnn_hidden, num_layers, n)
    return best


def search(target_total=60_550, enc_frac_target=ENC_FRAC_TARGET, gnn_frac_target=GNN_FRAC_TARGET,
           num_classes=6, tabular_dim=0, verbose=True):
    enc_target = round(enc_frac_target * target_total)
    gnn_target = round(gnn_frac_target * target_total)
    if verbose:
        print(f"Targets: total={target_total:,}  encoder~{enc_target:,} ({100*enc_frac_target:.1f}%)  "
              f"gnn~{gnn_target:,} ({100*gnn_frac_target:.1f}%)\n")

    c0, h, o, enc_n = search_encoder(enc_target)
    if verbose:
        print(f"Best encoder config: stem_channels={c0}, hidden_dim={h}, output_dim={o} "
              f"-> {enc_n:,} params (target {enc_target:,})")

    node_dim, edge_dim, gnn_hidden, num_layers, gnn_n = search_gnn(gnn_target)
    if verbose:
        print(f"Best GNN config: node_dim={node_dim}, edge_dim={edge_dim}, "
              f"gnn_hidden_dim={gnn_hidden}, num_layers={num_layers} -> {gnn_n:,} params "
              f"(target {gnn_target:,})")

    # encoder output size has to match the GNN edge size
    if o != edge_dim:
        if verbose:
            print(f"\nNote: encoder output_dim ({o}) != GNN edge_dim ({edge_dim}), "
                  f"re-searching encoder with output_dim pinned to {edge_dim}...")
        c0, h, o, enc_n = search_encoder(enc_target, output_dim_pin=edge_dim)
        if verbose:
            print(f"Re-solved encoder config: stem_channels={c0}, hidden_dim={h}, "
                  f"output_dim={o} -> {enc_n:,} params")

    pos_dim = SphericalHarmonicEncoding(max_degree=4).output_dim
    best_gp = None
    best_total_err = float("inf")
    for gp_hidden in range(1, 65):
        node_proj_n = count_params(nn.Linear(pos_dim, node_dim))
        graph_proj_n = count_params(nn.Sequential(nn.Linear(node_dim, gp_hidden), nn.Mish()))
        classifier_n = count_params(nn.Linear(gp_hidden + tabular_dim, num_classes))
        remainder = node_proj_n + graph_proj_n + classifier_n
        grand_total = enc_n + gnn_n + remainder
        err = abs(grand_total - target_total)
        if err < best_total_err:
            best_total_err = err
            best_gp = (gp_hidden, node_proj_n, graph_proj_n, classifier_n, grand_total)

    gp_hidden, node_proj_n, graph_proj_n, classifier_n, grand_total = best_gp

    config = dict(
        stem_channels=c0, encoder_hidden_dim=h, edge_dim=o, node_dim=node_dim,
        gnn_hidden_dim=gnn_hidden, num_layers=num_layers, graph_proj_hidden_dim=gp_hidden,
    )

    if verbose:
        print("\n=== FINAL CONFIG ===")
        for k, v in config.items():
            print(f"{k}={v}")
        print(f"\nGRAND TOTAL: {grand_total:,}  (target {target_total:,}, diff {grand_total - target_total:+,})")
        print(f"  encoder:    {enc_n:,} ({100*enc_n/grand_total:.1f}%)  [target ~{100*enc_frac_target:.1f}%]")
        print(f"  gnn_layers: {gnn_n:,} ({100*gnn_n/grand_total:.1f}%)  [target ~{100*gnn_frac_target:.1f}%]")
        print(f"  node_proj:  {node_proj_n:,}")
        print(f"  graph_proj: {graph_proj_n:,}")
        print(f"  classifier: {classifier_n:,}")

    return config, grand_total


def cmd_search(args):
    search(target_total=args.target_total, num_classes=args.num_classes)


def cmd_breakdown(args):

    def count_by_submodule(wrapper, label):
        print(f"\n=== {label} ===")
        wrapper.eval()
        try:
            with torch.no_grad():
                wrapper(torch.zeros(2, 12, 2500))
        except Exception as e:
            print(f"  (dummy forward failed, counting params as-is: {e})")
        total = count_params(wrapper)
        print(f"TOTAL: {total:,}")
        inner = wrapper.model

        for name, module in inner.named_children():
            n = count_params(module)
            if total:
                pct = 100 * n / total
            else:
                pct = 0
            print(f"  {name:25s} {n:>10,}  ({pct:5.1f}%)")

        for name, module in inner.named_children():
            children = list(module.named_children())
            if children:
                print(f"    -- inside {name} --")
                for sub_name, sub_module in children:
                    n = count_params(sub_module)
                    if total:
                        pct = 100 * n / total
                    else:
                        pct = 0
                    print(f"      {sub_name:23s} {n:>10,}  ({pct:5.1f}%)")

    count_by_submodule(GraphECGWrapper(num_classes=args.num_classes), "graphecg (full-size)")
    count_by_submodule(GraphECGSquashedV2Wrapper(num_classes=args.num_classes), "graphecg_squashed_v2")


SUPERCLASS_LABELS = {"NORM", "STTC", "CD", "HYP"}
SUBCLASS_LABELS = {"ASMI", "IMI"}


def build_6class_cohort(ptbxl_root, max_samples=None, subsample_seed=42):
    df = pd.read_csv(os.path.join(ptbxl_root, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    scp_statements = pd.read_csv(os.path.join(ptbxl_root, "scp_statements.csv"), index_col=0)
    diagnostic_statements = scp_statements[scp_statements.diagnostic == 1]
    code_to_superclass = diagnostic_statements["diagnostic_class"].to_dict()

    def get_labels(scp_dict):
        labels = {}
        for name in LABEL_NAMES:
            labels[name] = 0
        for code in scp_dict.keys():
            if code in SUBCLASS_LABELS:
                labels[code] = 1
            superclass = code_to_superclass.get(code)
            if superclass in SUPERCLASS_LABELS:
                labels[superclass] = 1
        return pd.Series(labels)

    label_df = df["scp_codes"].apply(get_labels)
    df = pd.concat([df, label_df], axis=1)
    df = df[df["validated_by_human"] == 1]
    df = df[df[LABEL_NAMES].sum(axis=1) > 0]

    if max_samples is not None and max_samples < len(df):
        rng = np.random.RandomState(subsample_seed)
        keep_idx = rng.choice(df.index.values, size=max_samples, replace=False)
        df = df.loc[keep_idx]

    train_ids = df[df.strat_fold <= 8].index.values
    val_ids = df[df.strat_fold == 9].index.values
    test_ids = df[df.strat_fold == 10].index.values
    return df, train_ids, val_ids, test_ids


def load_signal_250hz(ptbxl_root, filename_hr):
    record_path = os.path.join(ptbxl_root, filename_hr)
    signal, fields = wfdb.rdsamp(record_path)
    signal = sig.resample_poly(signal, up=250, down=500)
    return signal.T.astype(np.float32)


class ECG6ClassDataset(Dataset):
    def __init__(self, df, ecg_ids, ptbxl_root):
        self.df = df
        self.ecg_ids = ecg_ids
        self.ptbxl_root = ptbxl_root

    def __len__(self):
        return len(self.ecg_ids)

    def __getitem__(self, idx):
        ecg_id = self.ecg_ids[idx]
        row = self.df.loc[ecg_id]
        ecg = load_signal_250hz(self.ptbxl_root, row["filename_hr"])
        labels = torch.tensor(row[LABEL_NAMES].values.astype(np.float32))
        return torch.tensor(ecg), labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    all_true = []
    all_prob = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy()
            all_prob.append(prob)
            all_true.append(y.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    per_class_auc = {}
    per_class_f1 = {}
    for i, name in enumerate(LABEL_NAMES):
        try:
            per_class_auc[name] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            per_class_auc[name] = float("nan")
        per_class_f1[name] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)

    macro_auc = np.nanmean(list(per_class_auc.values()))
    macro_f1 = np.mean(list(per_class_f1.values()))
    return {"macro_auc": macro_auc, "macro_f1": macro_f1,
            "per_class_auc": per_class_auc, "per_class_f1": per_class_f1}


def cmd_train(args):
    data_dir = os.environ["DATASET_LOCATION"]
    save_root = os.environ.get("SAVE_LOCATION", ".")
    save_dir = os.path.join(save_root, "graphecg_squash_ptbxl6class", args.model)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(data_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

    df, train_ids, val_ids, test_ids = build_6class_cohort(path, max_samples=args.max_samples)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.model} seed {args.seed}] Device: {device}")

    train_ds = ECG6ClassDataset(df, train_ids, path)
    val_ds = ECG6ClassDataset(df, val_ids, path)
    test_ds = ECG6ClassDataset(df, test_ids, path)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[{args.model} seed {args.seed}] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    model_class = MODEL_MAP[args.model]
    model = model_class(num_classes=len(LABEL_NAMES)).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.model} seed {args.seed}] Parameters: {num_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_path = os.path.join(save_dir, f"best_model_seed{args.seed}.pt")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)

        val_metrics = evaluate(model, val_loader, device)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[{args.model} seed {args.seed}] Epoch {epoch+1:03d}  "
                  f"train_loss={train_loss:.4f}  val_macro_auc={val_metrics['macro_auc']:.4f}  "
                  f"val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_auc"] > best_val_auc:
            # float() so newer PyTorch can load the checkpoint
            best_val_auc = float(val_metrics["macro_auc"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_auc": best_val_auc}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n[{args.model} seed {args.seed}] === TEST RESULTS (best val checkpoint, epoch {checkpoint['epoch']}) ===")
    print(f"  Macro AUC: {test_metrics['macro_auc']:.4f}")
    print(f"  Macro F1:  {test_metrics['macro_f1']:.4f}")

    results_to_save = {
        "macro_auc": test_metrics["macro_auc"],
        "macro_f1": test_metrics["macro_f1"],
        "per_class_auc": test_metrics["per_class_auc"],
        "per_class_f1": test_metrics["per_class_f1"],
    }
    with open(os.path.join(save_dir, f"test_results_seed{args.seed}.json"), "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"[{args.model} seed {args.seed}] Saved results to {save_dir}")


def cmd_aggregate(args):
    rows = []
    for model_name in MODEL_MAP.keys():
        for seed in args.seeds:
            path = os.path.join(args.results_dir, model_name, f"test_results_seed{seed}.json")
            if not os.path.exists(path):
                print(f"MISSING: {path}")
                continue
            with open(path) as f:
                results = json.load(f)
            row = {"model": model_name, "seed": seed,
                   "macro_auc": results["macro_auc"], "macro_f1": results["macro_f1"]}
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Per-seed results:")
    print(df.to_string(index=False))

    summary = df.groupby("model")[["macro_auc", "macro_f1"]].agg(["mean", "std"])
    summary_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print("\nMean +/- std across seeds:")
    print(summary.to_string())
    print(f"\nSaved to {args.out} and {summary_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    search_ap = sub.add_parser("search", help="Find squashed settings for a target number of params")
    search_ap.add_argument("--target_total", type=int, default=60_550)
    search_ap.add_argument("--num_classes", type=int, default=6)
    search_ap.set_defaults(func=cmd_search)

    breakdown_ap = sub.add_parser("breakdown", help="Show how many params each part has (full vs squashed)")
    breakdown_ap.add_argument("--num_classes", type=int, default=6)
    breakdown_ap.set_defaults(func=cmd_breakdown)

    train_ap = sub.add_parser("train", help="Train one model with one seed on PTB-XL 6-class")
    train_ap.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    train_ap.add_argument("--seed", type=int, required=True)
    train_ap.add_argument("--epochs", type=int, default=50)
    train_ap.add_argument("--batch_size", type=int, default=32)
    train_ap.add_argument("--lr", type=float, default=1e-3)
    train_ap.add_argument("--max_samples", type=int, default=None)
    train_ap.set_defaults(func=cmd_train)

    agg_ap = sub.add_parser("aggregate", help="Combine all the trained models/seeds into one CSV")
    agg_ap.add_argument("--results_dir", required=True)
    agg_ap.add_argument("--out", default="comparison_graphecg_squash.csv")
    agg_ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    agg_ap.set_defaults(func=cmd_aggregate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
