"""
Does training GraphECG with random lead dropout help it handle missing leads
at test time? PTB-XL, 3 classes (NORM / IMI / ASMI).

Trains a baseline (always 12 leads) and a dropout version (sometimes 6-12 leads),
then tests both with 12/9/6 leads two ways: subgraph (leads removed from the graph)
and zerofill (leads set to 0, graph stays full size).

Run:
  python graphecg_dropout_experiment.py train --mode baseline --seed 42
  python graphecg_dropout_experiment.py train --mode dropout --seed 42
  python graphecg_dropout_experiment.py eval --ckpt_mode baseline --method subgraph --out_dir $SAVE_LOCATION/lead_corruption_results
  (run eval for each baseline/dropout x subgraph/zerofill combo)

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
import wfdb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score


MI_LABELS = ["IMI", "ASMI", "NORM"]


def extract_disease_label(scp_dict, allowed_labels):
    for code in scp_dict.keys():
        if code in allowed_labels:
            return code
    return None


def match_norm_to_imi_in_fold(df_norm, df_imi, fold_number, seed):
    imi_fold = df_imi[df_imi["strat_fold"] == fold_number]
    norm_fold = df_norm[df_norm["strat_fold"] == fold_number]
    matched_ids = []
    for (sex, age_group), imi_group in imi_fold.groupby(["sex", "age_group"]):
        norm_candidates = norm_fold[(norm_fold["sex"] == sex) & (norm_fold["age_group"] == age_group)]
        sample_size = min(len(imi_group), len(norm_candidates))
        matched = norm_candidates.sample(n=sample_size, random_state=seed)
        matched_ids.extend(matched.index.tolist())
    return matched_ids


def build_mi_cohort(ptbxl_root, fix_cohort_seed=42):
    df = pd.read_csv(os.path.join(ptbxl_root, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    df["disease_label"] = df["scp_codes"].apply(lambda x: extract_disease_label(x, MI_LABELS))
    df = df[df["disease_label"].notnull()]
    df = df[df["validated_by_human"] == 1]
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 45, 60, 120], labels=["<30", "30-45", "45-60", "60+"])

    df_norm = df[df["disease_label"] == "NORM"]
    df_mi = df[df["disease_label"] != "NORM"]
    df_imi = df[df["disease_label"] == "IMI"]

    # match NORM to IMI by sex and age group in each fold
    matched_norm_ids = []
    for fold in range(1, 11):
        fold_ids = match_norm_to_imi_in_fold(df_norm, df_imi, fold, seed=fix_cohort_seed)
        matched_norm_ids.extend(fold_ids)

    df_norm_matched = df_norm.loc[matched_norm_ids]
    df_new = pd.concat([df_mi, df_norm_matched])
    df_new["disease_label"] = df_new["disease_label"].astype(str)

    # folds 1-8 = train, 9 = val, 10 = test
    train_ids = df_new[df_new.strat_fold <= 8].index.values
    val_ids = df_new[df_new.strat_fold == 9].index.values
    test_ids = df_new[df_new.strat_fold == 10].index.values
    return df_new, train_ids, val_ids, test_ids


def audit_cohort(df_new, train_ids, val_ids, test_ids, out_path=None, model_name="shared_cohort"):
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    patient_sets = {}
    report = {"model": model_name, "splits": {}}

    for name, ecg_ids in splits.items():
        sub = df_new.loc[ecg_ids]
        patients = set(sub["patient_id"].unique())
        patient_sets[name] = patients
        n_ecgs = len(ecg_ids)
        n_patients = len(patients)
        prevalence = sub["disease_label"].value_counts().to_dict()
        prevalence_pct = {}
        for k, v in prevalence.items():
            prevalence_pct[k] = round(100 * v / n_ecgs, 2)

        print(f"\n[{model_name}] === {name.upper()} ===")
        print(f"  ECGs: {n_ecgs}  Patients: {n_patients}")
        print(f"  Class counts: {prevalence}")
        print(f"  Class prevalence (%): {prevalence_pct}")

        report["splits"][name] = {"n_ecgs": n_ecgs, "n_patients": n_patients,
                                  "class_counts": prevalence, "class_prevalence_pct": prevalence_pct}

    overlap_tv = patient_sets["train"] & patient_sets["val"]
    overlap_tt = patient_sets["train"] & patient_sets["test"]
    overlap_vt = patient_sets["val"] & patient_sets["test"]
    leakage_found = bool(overlap_tv or overlap_tt or overlap_vt)
    report["leakage_check"] = {"train_val_overlap_patients": len(overlap_tv),
                               "train_test_overlap_patients": len(overlap_tt),
                               "val_test_overlap_patients": len(overlap_vt),
                               "leakage_found": leakage_found}
    print(f"\n[{model_name}] Leakage check: train/val={len(overlap_tv)}, "
          f"train/test={len(overlap_tt)}, val/test={len(overlap_vt)}")
    if leakage_found:
        print("  *** WARNING: PATIENT LEAKAGE DETECTED ***")
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

LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LIMB_INDICES = [0, 1, 2, 3, 4, 5]
GRAPHECG_LABEL_MAP = {"NORM": 0, "IMI": 1, "ASMI": 2}


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
                 num_layers=3, tabular_dim=0, num_classes=3, dropout=0.5):
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cmd_train(args):
    data_dir = os.environ.get("DATASET_LOCATION")
    save_root = os.environ.get("SAVE_LOCATION", ".")
    is_dropout = (args.mode == "dropout")
    if is_dropout:
        save_dir = os.path.join(save_root, "MI_res_graphecg_dropout_seeded")
    else:
        save_dir = os.path.join(save_root, "MI_res_graphecg_seeded")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(data_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

    df_new, train_ids, val_ids, test_ids = build_mi_cohort(path, fix_cohort_seed=42)
    audit_cohort(df_new, train_ids, val_ids, test_ids,
                 out_path=os.path.join(save_dir, f"cohort_audit_{args.mode}_seed{args.seed}.json"),
                 model_name=f"GraphECG_{args.mode}_seed{args.seed}")

    set_seed(args.seed)
    builder = ECGGraphBuilder()

    def build_split(ecg_ids, apply_dropout):
        graphs = []
        for ecg_id in ecg_ids:
            row = df_new.loc[ecg_id]
            ecg = load_signal_250hz(path, row["filename_hr"])

            if apply_dropout and random.random() < args.lead_dropout_prob:
                num_leads = random.randint(args.min_leads, args.max_leads)
                observed_idx = sorted(random.sample(range(12), num_leads))
            else:
                observed_idx = list(range(12))

            g = builder.build_from_array(ecg, lead_indices=observed_idx, bidirectional=True)
            g.y = torch.tensor([GRAPHECG_LABEL_MAP[row["disease_label"]]], dtype=torch.long)
            g.ecg_id = torch.tensor([int(ecg_id)], dtype=torch.long)
            graphs.append(g)
        return graphs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_dropout:
        # dropout graphs are random each run, so no caching
        print(f"[seed {args.seed}] Building TRAIN graphs (with {args.lead_dropout_prob:.0%} lead dropout, "
              f"{args.min_leads}-{args.max_leads} leads when applied)...")
        train_graphs = build_split(train_ids, apply_dropout=True)
        print(f"[seed {args.seed}] Building VAL graphs (full 12-lead, no dropout)...")
        val_graphs = build_split(val_ids, apply_dropout=False)
        retrain_graphs = train_graphs + val_graphs
        print(f"[seed {args.seed}] Train: {len(train_graphs)}  Val: {len(val_graphs)}")
        test_graphs = None
    else:
        # same cohort for every seed, so cache the graphs after the first run
        cache_dir = os.path.join(save_root, "graph_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "graphecg_mi_graphs.pt")
        if os.path.exists(cache_path):
            print(f"[seed {args.seed}] Loading cached graphs from {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            train_graphs = cached["train"]
            val_graphs = cached["val"]
            test_graphs = cached["test"]
        else:
            print(f"[seed {args.seed}] No cache found -- building graphs from raw signals "
                  f"(reads+resamples every file; later seeds will reuse this)...")
            train_graphs = build_split(train_ids, apply_dropout=False)
            val_graphs = build_split(val_ids, apply_dropout=False)
            test_graphs = build_split(test_ids, apply_dropout=False)
            torch.save({"train": train_graphs, "val": val_graphs, "test": test_graphs}, cache_path)
            print(f"[seed {args.seed}] Cached graphs to {cache_path} for reuse by other seeds")
        retrain_graphs = train_graphs + val_graphs
        print(f"[seed {args.seed}] Train: {len(train_graphs)}  Val: {len(val_graphs)}  Test: {len(test_graphs)}")

    retrain_loader = DataLoader(retrain_graphs, batch_size=args.batch_size, shuffle=True)
    model = GraphECG(num_classes=3, tabular_dim=0).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[seed {args.seed}] Parameters: {num_params:,}")

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0
        for batch in retrain_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)["logits"]
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            correct += (out.argmax(dim=-1) == batch.y).sum().item()
            n += batch.num_graphs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[seed {args.seed}] Epoch {epoch+1:03d}  Loss: {total_loss/n:.4f}  Acc: {correct/n:.4f}")

    if is_dropout:
        ckpt_name = f"graphecg_dropout_seed{args.seed}_b{args.batch_size}_lr{args.learning_rate}_e{args.epochs}.pt"
    else:
        ckpt_name = f"graphecg_seed{args.seed}_b{args.batch_size}_lr{args.learning_rate}_e{args.epochs}.pt"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    ckpt_payload = {"model": model.state_dict(), "seed": args.seed, "epoch": args.epochs}
    if is_dropout:
        ckpt_payload["lead_dropout_prob"] = args.lead_dropout_prob
        ckpt_payload["min_leads"] = args.min_leads
        ckpt_payload["max_leads"] = args.max_leads
    torch.save(ckpt_payload, ckpt_path)
    print(f"[seed {args.seed}] Saved checkpoint to {ckpt_path}")

    if not is_dropout:
        test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
        model.eval()
        y_true = []
        y_pred = []
        y_prob = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)["logits"]
                prob = torch.softmax(out, dim=-1).cpu().numpy()
                y_prob.append(prob[0])
                y_pred.append(int(out.argmax(dim=-1).item()))
                y_true.append(int(batch.y.item()))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        test_accuracy = accuracy_score(y_true, y_pred)
        res_dir = os.path.join(save_dir, f"test_results_seed{args.seed}")
        os.makedirs(res_dir, exist_ok=True)
        np.save(os.path.join(res_dir, "y_true.npy"), y_true)
        np.save(os.path.join(res_dir, "y_pred.npy"), y_pred)
        np.save(os.path.join(res_dir, "y_prob.npy"), np.array(y_prob))
        print(f"[seed {args.seed}] Test accuracy (full 12-lead): {test_accuracy:.4f}  -- results saved to {res_dir}")
    else:
        print(f"[seed {args.seed}] Note: no held-out test eval here -- use the `eval` subcommand "
              f"pointed at this checkpoint dir for the reduced-lead comparison.")


def get_lead_subsets(n_leads, n_trials, seed=123):
    # fixed seed so every model gets tested on the same lead subsets
    rng = random.Random(seed)
    subsets = []
    if n_leads == 6:
        subsets.append(list(LIMB_INDICES))
        remaining_trials = n_trials - 1
    else:
        remaining_trials = n_trials
    all_leads = list(range(12))
    for _ in range(remaining_trials):
        subsets.append(sorted(rng.sample(all_leads, n_leads)))
    return subsets


def build_zerofilled_graph(builder, ecg_12xT, observed_leads_idx):
    # keeps the full 12-lead graph but zeros out the missing leads
    observed_set = set(observed_leads_idx)
    ecg_zerofilled = ecg_12xT.copy()
    for lead_idx in range(12):
        if lead_idx not in observed_set:
            ecg_zerofilled[lead_idx, :] = 0.0
    return builder.build_from_array(ecg_zerofilled, bidirectional=True)


def _evaluate_metrics(y_true, y_pred, y_prob):
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    try:
        row["auc_ovr_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        row["auc_ovr_macro"] = float("nan")
    return row


def cmd_eval(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = os.environ["DATASET_LOCATION"]
    path = os.path.join(data_dir, "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

    print("Building cohort (same 3-class MI cohort used for training)...")
    df_new, train_ids, val_ids, test_ids = build_mi_cohort(path, fix_cohort_seed=42)
    print(f"Test set: {len(test_ids)} ECGs")

    print("Loading raw test signals once...")
    test_signals = {}
    test_labels = {}
    for ecg_id in test_ids:
        row = df_new.loc[ecg_id]
        test_signals[ecg_id] = load_signal_250hz(path, row["filename_hr"])
        test_labels[ecg_id] = GRAPHECG_LABEL_MAP[row["disease_label"]]
    print(f"Loaded {len(test_signals)} signals.")

    is_dropout = (args.ckpt_mode == "dropout")
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    elif is_dropout:
        ckpt_dir = os.path.join(os.environ.get("SAVE_LOCATION", "."), "MI_res_graphecg_dropout_seeded")
    else:
        ckpt_dir = os.path.join(os.environ.get("SAVE_LOCATION", "."), "MI_res_graphecg_seeded")

    if is_dropout:
        model_label = f"GraphECG_dropout_{args.method}"
    else:
        model_label = f"GraphECG_{args.method}"

    builder = ECGGraphBuilder()
    lead_counts = [12, 9, 6]
    results = []

    for lead_count in lead_counts:
        if lead_count == 12:
            trial_subsets = [list(range(12))]
        else:
            trial_subsets = get_lead_subsets(lead_count, args.n_trials)

        for trial_idx, observed_leads in enumerate(trial_subsets):
            missing = []
            for i in range(12):
                if i not in observed_leads:
                    missing.append(LEAD_ORDER[i])
            print(f"\n=== {model_label}  lead_count={lead_count}  trial={trial_idx}  missing={missing} ===")

            for seed in args.seeds:
                if is_dropout:
                    ckpt_name = f"graphecg_dropout_seed{seed}_b32_lr0.001_e150.pt"
                else:
                    ckpt_name = f"graphecg_seed{seed}_b32_lr0.001_e150.pt"
                ckpt = os.path.join(ckpt_dir, ckpt_name)
                model = GraphECG(num_classes=3, tabular_dim=0).to(device)
                checkpoint = torch.load(ckpt, map_location=device)
                model.load_state_dict(checkpoint["model"])
                model.eval()

                y_true = []
                y_pred = []
                y_prob = []
                with torch.no_grad():
                    for ecg_id in test_ids:
                        ecg = test_signals[ecg_id]
                        label = test_labels[ecg_id]
                        if args.method == "subgraph":
                            graph = builder.build_from_array(ecg, lead_indices=observed_leads, bidirectional=True)
                        else:
                            graph = build_zerofilled_graph(builder, ecg, observed_leads)
                        batch = Batch.from_data_list([graph]).to(device)
                        logits = model(batch)["logits"]
                        prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                        y_prob.append(prob)
                        y_pred.append(int(np.argmax(prob)))
                        y_true.append(label)

                metrics = _evaluate_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
                metrics["model"] = model_label
                metrics["seed"] = seed
                metrics["lead_count"] = lead_count
                metrics["trial"] = trial_idx
                metrics["n_observed_leads"] = len(observed_leads)
                results.append(metrics)
                print(f"  seed={seed}  acc={metrics['accuracy']:.3f}  auc={metrics['auc_ovr_macro']:.3f}")

    df = pd.DataFrame(results)
    if is_dropout:
        tag = f"dropout_{args.method}"
    else:
        tag = f"baseline_{args.method}"
    raw_path = os.path.join(args.out_dir, f"graphecg_{tag}_results_raw.csv")
    df.to_csv(raw_path, index=False)

    metric_cols = ["accuracy", "f1_macro", "mcc", "auc_ovr_macro"]
    trial_avg = df.groupby(["model", "seed", "lead_count"])[metric_cols].mean().reset_index()
    summary = trial_avg.groupby(["model", "lead_count"])[metric_cols].agg(["mean", "std"])
    summary_path = os.path.join(args.out_dir, f"graphecg_{tag}_summary.csv")
    summary.to_csv(summary_path)

    print(f"\n=== SUMMARY: {model_label} (mean +/- std across seeds, averaged over trials) ===")
    print(summary.to_string())
    print(f"\nSaved raw results to {raw_path}")
    print(f"Saved summary to {summary_path}")
    print("\nRun the other (checkpoint_mode, method) combinations and compare their _summary.csv "
          "files for the full before/after, subgraph-vs-zerofill picture.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    train_ap = sub.add_parser("train", help="Train the baseline or the dropout version of GraphECG")
    train_ap.add_argument("--mode", required=True, choices=["baseline", "dropout"])
    train_ap.add_argument("--seed", type=int, required=True)
    train_ap.add_argument("-e", "--epochs", type=int, default=150)
    train_ap.add_argument("-b", "--batch_size", type=int, default=32)
    train_ap.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    train_ap.add_argument("--lead_dropout_prob", type=float, default=0.5,
                          help="(dropout mode only) chance that a training ECG has some leads removed")
    train_ap.add_argument("--min_leads", type=int, default=6, help="(dropout mode only)")
    train_ap.add_argument("--max_leads", type=int, default=12, help="(dropout mode only)")
    train_ap.set_defaults(func=cmd_train)

    eval_ap = sub.add_parser("eval", help="Test one (checkpoint type, method) combo with fewer leads")
    eval_ap.add_argument("--ckpt_mode", required=True, choices=["baseline", "dropout"],
                         help="Which trained models to load")
    eval_ap.add_argument("--method", required=True, choices=["subgraph", "zerofill"],
                         help="How to deal with the missing leads when testing")
    eval_ap.add_argument("--ckpt_dir", default=None,
                         help="Use a different checkpoint folder (default is based on --ckpt_mode and SAVE_LOCATION)")
    eval_ap.add_argument("--out_dir", required=True)
    eval_ap.add_argument("--n_trials", type=int, default=5)
    eval_ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    eval_ap.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
