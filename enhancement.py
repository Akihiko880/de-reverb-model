import os
import glob
import time
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from torchaudio import load, save

# =========================
# FORCE GPU SETUP
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)

print(f"[INFO] Using device: {device}")

# =========================
# SAFE GLOBALS (FIX LIGHTNING 2.6 + TORCH 2.6)
# =========================
try:
    import torch.serialization
    from sgmse.data_module import SpecsDataModule
    torch.serialization.add_safe_globals([SpecsDataModule])
except Exception as e:
    print("[WARN] safe_globals not applied:", e)

# =========================
# MODEL IMPORTS
# =========================
from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import (
    StochasticRegenerationModel,
    ScoreModel,
    DiscriminativeModel
)
from sgmse.util.other import *

# optional logging tool
try:
    from pypapi import events, papi_high as high
except ImportError:
    high = None

# =========================
# ARGS
# =========================
parser = ArgumentParser()

parser.add_argument("--test_dir", type=str, required=True)
parser.add_argument("--enhanced_dir", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--mode", type=str, required=True,
                    choices=["score-only", "denoiser-only", "storm"])

parser.add_argument("--corrector", type=str, default="ald",
                    choices=["ald", "langevin", "none"])
parser.add_argument("--corrector-steps", type=int, default=1)
parser.add_argument("--snr", type=float, default=0.5)
parser.add_argument("--N", type=int, default=50)

args = parser.parse_args()

os.makedirs(args.enhanced_dir, exist_ok=True)

# =========================
# CONFIG
# =========================
model_sr = 16000
ckpt_path = args.ckpt

# =========================
# MODEL SELECTION
# =========================
if args.mode == "storm":
    model_cls = StochasticRegenerationModel
elif args.mode == "score-only":
    model_cls = ScoreModel
else:
    model_cls = DiscriminativeModel

# =========================
# LOAD CHECKPOINT (FIX TORCH 2.6)
# =========================
model = model_cls.load_from_checkpoint(
    ckpt_path,
    map_location=device,
    strict=False
)

model.eval(no_ema=False)
model = model.to(device)

print("[INFO] Model loaded successfully")

# =========================
# INFERENCE LOOP
# =========================
noisy_files = sorted(glob.glob(os.path.join(args.test_dir, "*.wav")))

for f in tqdm(noisy_files):

    y, sr = load(f)

    if sr != model_sr:
        raise ValueError(f"Sample rate mismatch: {sr} != {model_sr}")

    y = y.to(device)

    with torch.no_grad():
        x_hat = model.enhance(
            y,
            corrector=args.corrector,
            N=args.N,
            corrector_steps=args.corrector_steps,
            snr=args.snr
        )

    out_path = os.path.join(args.enhanced_dir, os.path.basename(f))

    save(
        out_path,
        x_hat.detach().cpu().squeeze().unsqueeze(0),
        model_sr
    )

print("[DONE] Enhancement finished")