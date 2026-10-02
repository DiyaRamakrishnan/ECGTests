"""
Compares 5 ECG models (conv_rgnn, st_rege, ribeiro, graphecg, xgnn4mi) on PTB-XL
6-class multi-label (NORM / STTC / CD / HYP / ASMI / IMI).

Run:
  python ptbxl_6class_benchmark.py train --model conv_rgnn --seed 42 --epochs 50
  python ptbxl_6class_benchmark.py aggregate --results_dir $SAVE_LOCATION/new_graph_models_6class

Needs DATASET_LOCATION (PTB-XL folder) and SAVE_LOCATION (output folder).
"""
import os
import ast
import json
import math
import argparse
import random

import numpy as np
import pandas as pd
import scipy.signal as sig
import networkx as nx
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
        print(f"Subsampled 6-class cohort: {len(df)} ECGs (from full cohort, seed={subsample_seed})")

    # folds 1-8 = train, 9 = val, 10 = test
    train_ids = df[df.strat_fold <= 8].index.values
    val_ids = df[df.strat_fold == 9].index.values
    test_ids = df[df.strat_fold == 10].index.values

    return df, train_ids, val_ids, test_ids


def audit_6class_cohort(df, train_ids, val_ids, test_ids, out_path=None, model_name="shared_cohort"):
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    patient_sets = {}
    report = {"model": model_name, "splits": {}}

    for name, ecg_ids in splits.items():
        subset = df.loc[ecg_ids]
        patients = set(subset["patient_id"].unique())
        patient_sets[name] = patients
        label_counts = {}
        for lbl in LABEL_NAMES:
            label_counts[lbl] = int(subset[lbl].sum())
        label_pct = {}
        for lbl, c in label_counts.items():
            label_pct[lbl] = round(100 * c / len(ecg_ids), 2)

        print(f"\n[{model_name}] === {name.upper()} ===")
        print(f"  ECGs: {len(ecg_ids)}  Patients: {len(patients)}")
        print(f"  Label counts (multi-label, sums can exceed n_ecgs): {label_counts}")
        print(f"  Label prevalence (%): {label_pct}")

        report["splits"][name] = {"n_ecgs": len(ecg_ids), "n_patients": len(patients),
                                  "label_counts": label_counts, "label_prevalence_pct": label_pct}

    overlap_tv = patient_sets["train"] & patient_sets["val"]
    overlap_tt = patient_sets["train"] & patient_sets["test"]
    overlap_vt = patient_sets["val"] & patient_sets["test"]
    leakage_found = bool(overlap_tv or overlap_tt or overlap_vt)
    report["leakage_check"] = {"train_val": len(overlap_tv), "train_test": len(overlap_tt),
                               "val_test": len(overlap_vt), "leakage_found": leakage_found}
    print(f"\n[{model_name}] Leakage check: train/val={len(overlap_tv)}, "
          f"train/test={len(overlap_tt)}, val/test={len(overlap_vt)}")
    if leakage_found:
        print("  *** WARNING: LEAKAGE DETECTED ***")
    else:
        print("  OK -- no leakage.")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Audit written to {out_path}")

    assert not leakage_found, "Same patient is in more than one split -- fix this before training."
    return report


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
                 num_layers=3, tabular_dim=7, num_classes=1, dropout=0.5):
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


LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NUM_PATCHES = 25


class GCN_25(nn.Module):
    def __init__(self, dataset, num_nodes, num_patches, num_classes):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_patches = num_patches
        self.num_classes = num_classes

        self.ffn1 = nn.Sequential(
            nn.Linear(100, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(400, 100), nn.BatchNorm1d(100), nn.Dropout(p=0.5))

        self.conv1 = GCNConv(dataset.num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.conv4 = GCNConv(16, 8)
        self.conv5 = GCNConv(8, 4)
        self.fc = nn.Linear(self.num_nodes * 4, self.num_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = x.float()
        x = self.ffn1(x)
        x = F.dropout(F.relu(self.conv1(x, edge_index)), training=self.training)
        x = F.dropout(F.relu(self.conv2(x, edge_index)), training=self.training)
        x = F.dropout(F.relu(self.conv3(x, edge_index)), training=self.training)
        x = F.dropout(F.relu(self.conv4(x, edge_index)), training=self.training)
        x = F.dropout(F.relu(self.conv5(x, edge_index)), training=self.training)
        x = torch.reshape(x, (-1, self.num_nodes * 4))
        return self.fc(x)


def _build_full_xgnn4mi_graph(ecg_12xT, num_patches=NUM_PATCHES):
    limb_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF']
    chest_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    specific_leads = ['I', 'AVF', 'V4', 'V5']

    G = nx.Graph()
    T = ecg_12xT.shape[1]
    patch_size = T // num_patches

    for i, lead in enumerate(LEAD_ORDER):
        signal_data = ecg_12xT[i]
        for j in range(num_patches):
            G.add_node(f"{lead}_p{j}", signal=signal_data[j * patch_size:(j + 1) * patch_size])

    for lead in LEAD_ORDER:
        for j in range(num_patches - 1):
            G.add_edge(f"{lead}_p{j}", f"{lead}_p{j+1}")

    for group in (limb_leads, chest_leads, specific_leads):
        for lead1 in group:
            for lead2 in group:
                if lead1 != lead2:
                    for j in range(num_patches):
                        G.add_edge(f"{lead1}_p{j}", f"{lead2}_p{j}")

    node_id_mapping = {}
    for i, node_name in enumerate(G.nodes()):
        node_id_mapping[node_name] = i
    x = []
    edge_index = []
    for node in G.nodes():
        x.append(G.nodes[node]['signal'])
        node_id = node_id_mapping[node]
        for neighbor in G.neighbors(node):
            edge_index.append([node_id, node_id_mapping[neighbor]])

    x = torch.tensor(np.array(x), dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


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


class XGNN4MIWrapper(nn.Module):
    def __init__(self, num_classes=6, num_patches=NUM_PATCHES):
        super().__init__()

        class _DummyDataset:
            num_features = 100

        self.model = GCN_25(_DummyDataset(), num_nodes=12 * num_patches,
                            num_patches=num_patches, num_classes=num_classes)
        self.num_patches = num_patches

    def forward(self, x_ecg):
        device = x_ecg.device
        x_np = x_ecg.detach().cpu().numpy()
        data_list = []
        for b in range(x_ecg.shape[0]):
            data_list.append(_build_full_xgnn4mi_graph(x_np[b], self.num_patches))
        batch = Batch.from_data_list(data_list).to(device)
        return self.model(batch)


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


# my reimplementation of Conv-RGNN (Qiang et al. 2024) -- the edges are a guess
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


# simplified ST-ReGE: uses whole leads as nodes instead of patches
class STReGE(nn.Module):
    def __init__(self, feat_dim=64, hidden_dim=64, num_classes=6):
        super().__init__()
        self.extractor = LeadFeatureExtractor(out_dim=feat_dim)
        self.temporal = nn.GRU(feat_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.temporal_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.spatial_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.spatial_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_ecg):
        device = x_ecg.device
        B = x_ecg.shape[0]
        feats = self.extractor(x_ecg)
        temporal_out, _ = self.temporal(feats)
        temporal_out = F.relu(self.temporal_proj(temporal_out))

        edges = []
        for i in range(12):
            for j in range(12):
                if i != j:
                    edges.append((i, j))
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
        data_list = []
        for b in range(B):
            data_list.append(Data(x=temporal_out[b], edge_index=edge_index))
        batch = Batch.from_data_list(data_list)

        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        residual = x
        x = F.relu(self.spatial_conv1(x, edge_index))
        x = self.spatial_conv2(x, edge_index) + residual
        return self.classifier(global_mean_pool(x, batch_idx))


class ResidualUnit(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size=16, dropout_rate=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size, stride=downsample, padding=pad, bias=False)
        if downsample > 1:
            self.skip_pool = nn.MaxPool1d(downsample, stride=downsample)
        else:
            self.skip_pool = nn.Identity()
        if n_filters_in != n_filters_out:
            self.skip_conv = nn.Conv1d(n_filters_in, n_filters_out, kernel_size=1, bias=False)
        else:
            self.skip_conv = nn.Identity()
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, y):
        skip = self.skip_conv(self.skip_pool(y))
        main = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        main = self.conv2(main)
        if skip.shape[-1] != main.shape[-1]:
            min_len = min(skip.shape[-1], main.shape[-1])
            skip = skip[..., :min_len]
            main = main[..., :min_len]
        x_out = main + skip
        y_out = x_out
        x_out = self.dropout2(F.relu(self.bn2(x_out)))
        return x_out, y_out


# port of antonior92/automatic-ecg-diagnosis
class RibeiroResNet1D(nn.Module):
    def __init__(self, num_classes=6, kernel_size=16, dropout_rate=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.stem_conv = nn.Conv1d(12, 64, kernel_size, padding=pad, bias=False)
        self.stem_bn = nn.BatchNorm1d(64)
        self.block1 = ResidualUnit(64, 128, downsample=4, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.block2 = ResidualUnit(128, 196, downsample=4, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.block3 = ResidualUnit(196, 256, downsample=4, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.block4 = ResidualUnit(256, 320, downsample=4, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x_ecg):
        x = F.relu(self.stem_bn(self.stem_conv(x_ecg)))
        y = x
        x, y = self.block1(x, y)
        x, y = self.block2(x, y)
        x, y = self.block3(x, y)
        x, _ = self.block4(x, y)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


MODEL_MAP = {
    "conv_rgnn": ConvRGNN,
    "st_rege": STReGE,
    "ribeiro": RibeiroResNet1D,
    "graphecg": GraphECGWrapper,
    "xgnn4mi": XGNN4MIWrapper,
}


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


def train(args):
    data_dir = os.environ["DATASET_LOCATION"]
    save_root = os.environ.get("SAVE_LOCATION", ".")
    save_dir = os.path.join(save_root, "new_graph_models_6class", args.model)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(data_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

    df, train_ids, val_ids, test_ids = build_6class_cohort(path, max_samples=args.max_samples)
    audit_6class_cohort(df, train_ids, val_ids, test_ids,
                        out_path=os.path.join(save_dir, f"cohort_audit_seed{args.seed}.json"),
                        model_name=f"{args.model}_seed{args.seed}")

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

    # Ribeiro's LazyLinear needs one forward pass before its params exist
    model.eval()
    with torch.no_grad():
        dummy_x = torch.zeros(2, 12, 2500, device=device)
        model(dummy_x)

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
    print(f"  Per-class AUC: {test_metrics['per_class_auc']}")
    print(f"  Per-class F1:  {test_metrics['per_class_f1']}")

    results_to_save = {
        "macro_auc": test_metrics["macro_auc"],
        "macro_f1": test_metrics["macro_f1"],
        "per_class_auc": test_metrics["per_class_auc"],
        "per_class_f1": test_metrics["per_class_f1"],
    }
    with open(os.path.join(save_dir, f"test_results_seed{args.seed}.json"), "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"[{args.model} seed {args.seed}] Saved results to {save_dir}")


def aggregate(args):
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
            for name in LABEL_NAMES:
                row[f"auc_{name}"] = results["per_class_auc"].get(name, float("nan"))
                row[f"f1_{name}"] = results["per_class_f1"].get(name, float("nan"))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Per-seed results:")
    print(df.to_string(index=False))

    summary_cols = ["macro_auc", "macro_f1"]
    for n in LABEL_NAMES:
        summary_cols.append(f"auc_{n}")
    summary = df.groupby("model")[summary_cols].agg(["mean", "std"])
    summary_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print("\nMean +/- std across seeds:")
    print(summary.to_string())
    print(f"\nSaved to {args.out} and {summary_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    train_ap = sub.add_parser("train", help="Train one model with one seed")
    train_ap.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    train_ap.add_argument("--seed", type=int, required=True)
    train_ap.add_argument("--epochs", type=int, default=50)
    train_ap.add_argument("--batch_size", type=int, default=32)
    train_ap.add_argument("--lr", type=float, default=1e-3)
    train_ap.add_argument("--max_samples", type=int, default=None,
                          help="If set, only use this many ECGs (picked randomly before splitting).")
    train_ap.set_defaults(func=train)

    agg_ap = sub.add_parser("aggregate", help="Combine all the trained models/seeds into one CSV")
    agg_ap.add_argument("--results_dir", required=True)
    agg_ap.add_argument("--out", default="comparison_ptbxl_6class.csv")
    agg_ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    agg_ap.set_defaults(func=aggregate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
