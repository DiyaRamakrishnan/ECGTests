"""
Compares 5 models (conv_rgnn, st_rege, ribeiro, graphecg, xgnn4mi) on EchoNext:
12-lead ECG + 7 clinical features in, structural heart disease yes/no out.
Only GraphECG uses the clinical features.

Run:
  python echonext_5way_benchmark.py extract   (downloads a random subset from PhysioNet)
  python echonext_5way_benchmark.py train --model conv_rgnn --seed 42 --epochs 150
  python echonext_5way_benchmark.py aggregate --results_dir $SAVE_LOCATION/echonext_5way

Needs SAVE_LOCATION (output folder). Change DATA_DIR below to your own folder.
"""
import os
import io
import json
import math
import time
import argparse
import random
import subprocess

import numpy as np
import numpy.lib.format as npformat
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch

from sklearn.metrics import roc_auc_score, f1_score


# CHANGE THIS to your own data folder
DATA_DIR = "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2026/projects/colon_st_diya_shreyas/interpretation/benchmark/columbia_baseline"


PHYSIONET_BASE = "https://physionet.org/files/echonext/1.1.1/"

# sized for a ~3 hour download at PhysioNet's speed limit
SPLIT_COUNTS = {"train": 6000, "val": 1200, "test": 1200}
NUM_CHUNKS = 30
SAMPLING_TAG = "stratified_random_chunks_v4"
SEED = 2026
ASSUMED_MIN_BYTES_PER_SEC = 80_000
MIN_CHUNK_TIMEOUT = 180


def wget_fetch_small(url, out_path, byte_range=None, max_retries=4, backoff_base=5, timeout=60):
    for attempt in range(1, max_retries + 1):
        cmd = ["wget", "--auth-no-challenge", "-q", "--tries=1", f"--timeout={timeout}"]
        if byte_range is not None:
            start, end = byte_range
            cmd += ["--header", f"Range: bytes={start}-{end}"]
        cmd += ["-O", out_path, url]
        result = subprocess.run(cmd)
        if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
        wait = backoff_base * attempt
        print(f"  [retry {attempt}/{max_retries}] wget failed (rc={result.returncode}), waiting {wait}s...", flush=True)
        time.sleep(wait)
    return False


def fetch_npy_header(url):
    tmp = "/tmp/_echonext_header_probe.bin"
    ok = wget_fetch_small(url, tmp, byte_range=(0, 8191))
    assert ok, f"Couldn't download the header for {url}"
    with open(tmp, "rb") as f:
        raw = f.read()
    bio = io.BytesIO(raw)
    version = npformat.read_magic(bio)
    shape, fortran_order, dtype = npformat._read_array_header(bio, version)
    data_offset = bio.tell()
    assert not fortran_order, f"Didn't expect fortran_order=True for {url}"
    os.remove(tmp)
    return shape, dtype, data_offset


def fetch_chunk_at_offset(url, start_byte, num_bytes, out_path, max_retries=4, backoff_base=5, timeout=300):
    one_mb = 1024 * 1024
    for attempt in range(1, max_retries + 1):
        # a hand-set Range header doesn't work for non-zero offsets, --start-pos does
        cmd = ["wget", "--auth-no-challenge", "-q", f"--start-pos={start_byte}", "-O", "-", url]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        data = bytearray()
        t0 = time.time()
        timed_out = False
        try:
            while len(data) < num_bytes:
                if time.time() - t0 > timeout:
                    timed_out = True
                    break
                chunk = proc.stdout.read(min(one_mb, num_bytes - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        if len(data) == num_bytes:
            with open(out_path, "wb") as f:
                f.write(bytes(data))
            return True

        if timed_out:
            reason = "timed out"
        else:
            reason = f"got {len(data)}/{num_bytes} bytes"
        wait = backoff_base * attempt
        print(f"    [retry {attempt}/{max_retries}] chunk fetch {reason}, waiting {wait}s...", flush=True)
        time.sleep(wait)
    return False


def _split_already_done(manifest, split_name, target_n):
    info = manifest.get(split_name)
    if not info:
        return False
    return info.get("subset_n") == target_n and info.get("sampling") == SAMPLING_TAG


def extract(args):
    os.makedirs(DATA_DIR, exist_ok=True)

    meta_path = os.path.join(DATA_DIR, "echonext_metadata_100k.csv")
    if not os.path.exists(meta_path):
        print("Fetching full metadata CSV (small file)...", flush=True)
        ok = wget_fetch_small(PHYSIONET_BASE + "echonext_metadata_100k.csv", meta_path, timeout=120)
        assert ok, "Couldn't download echonext_metadata_100k.csv"
    meta = pd.read_csv(meta_path)

    manifest_path = os.path.join(DATA_DIR, "echonext_subset_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    rng = np.random.default_rng(SEED)

    for split_name, target_n in SPLIT_COUNTS.items():
        if _split_already_done(manifest, split_name, target_n):
            print(f"[{split_name}] Already extracted at target_n={target_n} with {SAMPLING_TAG}, skipping.")
            continue

        split_meta = meta[meta["split"] == split_name].reset_index(drop=True)
        full_n = len(split_meta)

        rows_per_chunk = target_n // NUM_CHUNKS
        remainder = target_n - rows_per_chunk * NUM_CHUNKS
        bin_size = full_n // NUM_CHUNKS
        assert bin_size >= rows_per_chunk + 1, (
            f"[{split_name}] full_n={full_n} is too small for {NUM_CHUNKS} sections of "
            f"{rows_per_chunk} rows each -- lower NUM_CHUNKS or target_n.")

        wave_url = PHYSIONET_BASE + f"EchoNext_{split_name}_waveforms.npy"
        shape, dtype, data_offset = fetch_npy_header(wave_url)
        row_nbytes = dtype.itemsize
        for d in shape[1:]:
            row_nbytes *= d
        print(f"[{split_name}] full_n={full_n}  target_n={target_n}  "
              f"{NUM_CHUNKS} chunks x ~{rows_per_chunk} rows  (row_nbytes={row_nbytes})", flush=True)

        sampled_ranges = []
        chunk_arrays = []
        for i in range(NUM_CHUNKS):
            if i < remainder:
                this_chunk_rows = rows_per_chunk + 1
            else:
                this_chunk_rows = rows_per_chunk

            bin_start = i * bin_size
            if i < NUM_CHUNKS - 1:
                bin_end = (i + 1) * bin_size
            else:
                bin_end = full_n

            max_start = bin_end - this_chunk_rows
            row_start = int(rng.integers(bin_start, max_start + 1))
            row_end = row_start + this_chunk_rows

            byte_start = data_offset + row_start * row_nbytes
            n_bytes = this_chunk_rows * row_nbytes
            chunk_path = os.path.join(DATA_DIR, f"_chunk_{split_name}_{i}.bin")

            chunk_timeout = max(MIN_CHUNK_TIMEOUT, int(n_bytes / ASSUMED_MIN_BYTES_PER_SEC * 1.3))
            print(f"  [{split_name}] chunk {i + 1}/{NUM_CHUNKS}: rows {row_start}-{row_end} "
                  f"({n_bytes / 1e6:.1f} MB, timeout={chunk_timeout}s)...", flush=True)
            ok = fetch_chunk_at_offset(wave_url, byte_start, n_bytes, chunk_path, timeout=chunk_timeout)
            assert ok, f"[{split_name}] Couldn't download chunk {i} (rows {row_start}-{row_end})"

            with open(chunk_path, "rb") as f:
                raw_bytes = f.read()
            arr = np.frombuffer(raw_bytes, dtype=dtype).reshape((this_chunk_rows,) + tuple(shape[1:]))
            chunk_arrays.append(arr)
            sampled_ranges.append([row_start, row_end])
            os.remove(chunk_path)

        subset_waveforms = np.concatenate(chunk_arrays, axis=0)
        assert subset_waveforms.shape[0] == target_n

        out_path = os.path.join(DATA_DIR, f"EchoNext_{split_name}_subset_waveforms.npy")
        np.save(out_path, subset_waveforms)

        manifest[split_name] = {
            "waveform_file": os.path.basename(out_path),
            "subset_n": target_n,
            "full_n": full_n,
            "sampled_ranges": sampled_ranges,
            "sampling": SAMPLING_TAG,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[{split_name}] Done -- saved {target_n} rows sampled from "
              f"{NUM_CHUNKS} random chunks spanning all {full_n} rows.", flush=True)

    print("\nAll splits ready.")


LABEL_COL = "shd_moderate_or_greater_flag"

# no echo-derived columns, those would leak the label
TABULAR_COLS = [
    "age_at_ecg", "sex", "ventricular_rate", "atrial_rate",
    "pr_interval", "qrs_duration", "qt_corrected",
]

SEX_MAP = {
    "M": 1.0, "Male": 1.0, "male": 1.0, "MALE": 1.0, "1": 1.0, 1: 1.0,
    "F": 0.0, "Female": 0.0, "female": 0.0, "FEMALE": 0.0, "0": 0.0, 0: 0.0,
}


def _select_rows(split_meta, info, k):
    if "sampled_ranges" in info:
        indices = []
        for start, end in info["sampled_ranges"]:
            indices.extend(range(start, end))
        assert len(indices) == k, f"sampled_ranges cover {len(indices)} rows, expected {k}"
        return split_meta.iloc[indices].reset_index(drop=True)
    else:
        return split_meta.iloc[:k].reset_index(drop=True)


class EchoNextSubsetDataset(Dataset):

    def __init__(self, data_dir, split_name, tabular_stats=None):
        manifest_path = os.path.join(data_dir, "echonext_subset_manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        info = manifest[split_name]

        waveform_path = os.path.join(data_dir, info["waveform_file"])
        waveforms = np.load(waveform_path)
        k = waveforms.shape[0]
        assert k == info["subset_n"], f"[{split_name}] Loaded {k} samples but manifest says {info['subset_n']}"
        self.waveforms = waveforms[:, 0, :, :]

        metadata_path = os.path.join(data_dir, "echonext_metadata_100k.csv")
        meta = pd.read_csv(metadata_path)
        split_meta = meta[meta["split"] == split_name].reset_index(drop=True)
        assert len(split_meta) == info["full_n"], (
            f"[{split_name}] Metadata split='{split_name}' row count ({len(split_meta)}) "
            f"!= full waveform file's declared N ({info['full_n']}) -- the rows might not "
            f"line up. STOP and figure this out before trusting the labels.")

        rows = _select_rows(split_meta, info, k)
        assert len(rows) == k

        self.labels = rows[LABEL_COL].values.astype(np.float32)
        assert len(self.labels) == k

        tab_df = rows[TABULAR_COLS].copy()
        raw_sex = tab_df["sex"]
        tab_df["sex"] = raw_sex.map(SEX_MAP)
        n_unmapped = tab_df["sex"].isna().sum() - raw_sex.isna().sum()
        assert n_unmapped == 0, (
            f"[{split_name}] {n_unmapped} 'sex' values didn't match SEX_MAP -- "
            f"values found: {sorted(raw_sex.dropna().unique().tolist())}. "
            f"Add them to SEX_MAP before trusting this feature.")
        tab_arr = tab_df.to_numpy(dtype=np.float64)

        # no stats passed in means this is the train split
        if tabular_stats is None:
            mean = np.nanmean(tab_arr, axis=0)
            std = np.nanstd(tab_arr, axis=0)
            std[std == 0] = 1.0
        else:
            mean, std = tabular_stats
        self.tabular_stats = (mean, std)

        nan_mask = np.isnan(tab_arr)
        if nan_mask.any():
            tab_arr = np.where(nan_mask, mean[np.newaxis, :], tab_arr)
        tab_arr = (tab_arr - mean) / std
        self.tabular = tab_arr.astype(np.float32)

        n_pos = int(self.labels.sum())
        print(f"[{split_name}] {k} samples, {n_pos} SHD-positive ({100 * n_pos / k:.1f}%), "
              f"tabular NaNs imputed: {int(nan_mask.sum())}/{nan_mask.size}")

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        ecg = self.waveforms[idx].T
        ecg = np.ascontiguousarray(ecg, dtype=np.float32)
        tabular = torch.tensor(self.tabular[idx], dtype=torch.float32)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return torch.tensor(ecg), tabular, label


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


class GraphECGTabularWrapper(nn.Module):
    def __init__(self, num_classes=1, tabular_dim=7):
        super().__init__()
        self.model = GraphECG(num_classes=num_classes, tabular_dim=tabular_dim)
        self.builder = ECGGraphBuilder()

    def forward(self, x_ecg, tabular=None):
        device = x_ecg.device
        x_np = x_ecg.detach().cpu().numpy()
        data_list = []
        for b in range(x_ecg.shape[0]):
            data_list.append(self.builder.build_from_array(x_np[b], bidirectional=True))
        batch = Batch.from_data_list(data_list).to(device)
        return self.model(batch, tabular=tabular)["logits"]


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


class XGNN4MIWrapper(nn.Module):
    def __init__(self, num_classes=1, num_patches=NUM_PATCHES):
        super().__init__()

        class _DummyDataset:
            num_features = 100

        self.model = GCN_25(_DummyDataset(), num_nodes=12 * num_patches,
                            num_patches=num_patches, num_classes=num_classes)
        self.num_patches = num_patches

    def forward(self, x_ecg, tabular=None):
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


class ConvRGNN(nn.Module):
    def __init__(self, feat_dim=64, hidden_dim=64, num_classes=1, num_layers=3):
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

    def forward(self, x_ecg, tabular=None):
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


class STReGE(nn.Module):
    def __init__(self, feat_dim=64, hidden_dim=64, num_classes=1):
        super().__init__()
        self.extractor = LeadFeatureExtractor(out_dim=feat_dim)
        self.temporal = nn.GRU(feat_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.temporal_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.spatial_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.spatial_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_ecg, tabular=None):
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


class RibeiroResNet1D(nn.Module):
    def __init__(self, num_classes=1, kernel_size=16, dropout_rate=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.stem_conv = nn.Conv1d(12, 64, kernel_size, padding=pad, bias=False)
        self.stem_bn = nn.BatchNorm1d(64)
        self.block1 = ResidualUnit(64, 128, downsample=4, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.block2 = ResidualUnit(128, 196, downsample=4, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.block3 = ResidualUnit(196, 256, downsample=4, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.block4 = ResidualUnit(256, 320, downsample=4, kernel_size=kernel_size, dropout_rate=dropout_rate)
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x_ecg, tabular=None):
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
    "graphecg": GraphECGTabularWrapper,
    "xgnn4mi": XGNN4MIWrapper,
}
ECHONEXT_MODELS = list(MODEL_MAP.keys())
TABULAR_MODELS = {"graphecg"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def call_model(model, model_name, x, tabular, device):
    if model_name in TABULAR_MODELS:
        return model(x, tabular=tabular.to(device))
    return model(x)


def evaluate(model, model_name, loader, device):
    model.eval()
    all_true = []
    all_prob = []
    with torch.no_grad():
        for x, tabular, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = call_model(model, model_name, x, tabular, device)
            prob = torch.sigmoid(logits).cpu().numpy()
            all_prob.append(prob)
            all_true.append(y.cpu().numpy())
    y_true = np.concatenate(all_true).ravel()
    y_prob = np.concatenate(all_prob).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"auc": auc, "f1": f1}


def train(args):
    save_root = os.environ.get("SAVE_LOCATION", ".")
    save_dir = os.path.join(save_root, "echonext_5way", args.model)
    os.makedirs(save_dir, exist_ok=True)

    train_ds = EchoNextSubsetDataset(DATA_DIR, "train")
    tabular_stats = train_ds.tabular_stats
    val_ds = EchoNextSubsetDataset(DATA_DIR, "val", tabular_stats=tabular_stats)
    test_ds = EchoNextSubsetDataset(DATA_DIR, "test", tabular_stats=tabular_stats)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.model} seed {args.seed}] Device: {device}")
    print(f"[{args.model} seed {args.seed}] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_class = MODEL_MAP[args.model]
    model = model_class(num_classes=1).to(device)

    # Ribeiro's LazyLinear needs one forward pass before its params exist
    model.eval()
    with torch.no_grad():
        dummy_x = torch.zeros(2, 12, 2500, device=device)
        dummy_tab = torch.zeros(2, 7, device=device)
        call_model(model, args.model, dummy_x, dummy_tab, device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.model} seed {args.seed}] Parameters: {num_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_path = os.path.join(save_dir, f"best_model_seed{args.seed}.pt")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, tabular, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = call_model(model, args.model, x, tabular, device)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)

        val_metrics = evaluate(model, args.model, val_loader, device)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{args.model} seed {args.seed}] Epoch {epoch + 1:03d}  "
                  f"train_loss={train_loss:.4f}  val_auc={val_metrics['auc']:.4f}  "
                  f"val_f1={val_metrics['f1']:.4f}")

        if val_metrics["auc"] > best_val_auc:
            # float() so newer PyTorch can load the checkpoint
            best_val_auc = float(val_metrics["auc"])
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": best_val_auc}, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, args.model, test_loader, device)

    print(f"\n[{args.model} seed {args.seed}] === TEST RESULTS (best val checkpoint, epoch {checkpoint['epoch']}) ===")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  F1:  {test_metrics['f1']:.4f}")

    with open(os.path.join(save_dir, f"test_results_seed{args.seed}.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[{args.model} seed {args.seed}] Saved results to {save_dir}")


def aggregate(args):
    rows = []
    for model_name in ECHONEXT_MODELS:
        for seed in args.seeds:
            path = os.path.join(args.results_dir, model_name, f"test_results_seed{seed}.json")
            if not os.path.exists(path):
                print(f"MISSING: {path}")
                continue
            with open(path) as f:
                metrics = json.load(f)
            row = {"model": model_name, "seed": seed}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("Per-seed results:")
    print(df.to_string(index=False))

    summary = df.groupby("model")[["auc", "f1"]].agg(["mean", "std"])
    summary_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print("\nMean +/- std across seeds:")
    print(summary.to_string())
    print(f"\nSaved per-seed table to {args.out}")
    print(f"Saved summary table to {summary_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    extract_ap = sub.add_parser("extract", help="Download a random chunk of EchoNext from PhysioNet")
    extract_ap.set_defaults(func=extract)

    train_ap = sub.add_parser("train", help="Train one model with one seed")
    train_ap.add_argument("--model", required=True, choices=ECHONEXT_MODELS)
    train_ap.add_argument("--seed", type=int, required=True)
    train_ap.add_argument("--epochs", type=int, default=150)
    train_ap.add_argument("--batch_size", type=int, default=32)
    train_ap.add_argument("--lr", type=float, default=1e-3)
    train_ap.set_defaults(func=train)

    agg_ap = sub.add_parser("aggregate", help="Combine all the trained models/seeds into one CSV")
    agg_ap.add_argument("--results_dir", required=True)
    agg_ap.add_argument("--out", default="comparison_echonext_5way.csv")
    agg_ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    agg_ap.set_defaults(func=aggregate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
