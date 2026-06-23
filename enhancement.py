import os
import glob
import torch
import torch.multiprocessing as mp
import threading
import queue
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from torchaudio import load, save

# =========================
# GLOBAL CONFIG
# =========================
model_sr = 16000

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


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

# =========================
# SAFE GLOBALS (Torch 2.6 fix)
# =========================
try:
    import torch.serialization
    torch.serialization.add_safe_globals([SpecsDataModule])
except Exception as e:
    print("[WARN] safe_globals skipped:", e)


# =========================
# ARGPARSE
# =========================
parser = ArgumentParser()

parser.add_argument("--test_dir", type=str, required=True)
parser.add_argument("--enhanced_dir", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)

parser.add_argument("--mode", type=str, required=True,
                    choices=["score-only", "denoiser-only", "storm"])

parser.add_argument("--corrector", type=str, default="ald",
                    choices=["ald", "langevin", "none"])
parser.add_argument("--corrector_steps", type=int, default=1)
parser.add_argument("--snr", type=float, default=0.5)
parser.add_argument("--N", type=int, default=50)

args = parser.parse_args()


# =========================
# MODEL SELECTOR
# =========================
def get_model_class(mode):
    if mode == "storm":
        return StochasticRegenerationModel
    elif mode == "score-only":
        return ScoreModel
    else:
        return DiscriminativeModel


model_cls = get_model_class(args.mode)


# =========================
# PREFETCH WORKER (CPU I/O PIPELINE)
# =========================
def prefetch_worker(file_list, q):
    for f in file_list:
        try:
            y, sr = load(f)
            if sr != model_sr:
                continue
            q.put((f, y))
        except Exception as e:
            print(f"[LOAD ERROR] {f}: {e}")

    q.put(None)


# =========================
# GPU WORKER
# =========================
def gpu_worker(gpu_id, q, args):

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    print(f"[GPU {gpu_id}] loading model...")

    model = model_cls.load_from_checkpoint(
        args.ckpt,
        map_location=device,
        strict=False
    )

    model.eval(no_ema=False)
    model = model.to(device)

    print(f"[GPU {gpu_id}] ready")

    while True:

        item = q.get()
        if item is None:
            break

        f, y = item

        y = y.to(device, non_blocking=True)

        try:
            with torch.no_grad(), torch.cuda.amp.autocast():

                x_hat = model.enhance(
                    y,
                    corrector=args.corrector,
                    N=args.N,
                    corrector_steps=args.corrector_steps,
                    snr=args.snr
                )

            out_path = os.path.join(
                args.enhanced_dir,
                f"gpu{gpu_id}_" + os.path.basename(f)
            )

            save(
                out_path,
                x_hat.detach().cpu().squeeze().unsqueeze(0),
                model_sr
            )

        except Exception as e:
            print(f"[GPU {gpu_id} ERROR] {f}: {e}")

    print(f"[GPU {gpu_id}] finished")


# =========================
# MAIN PIPELINE
# =========================
def main(args):

    os.makedirs(args.enhanced_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.test_dir, "*.wav")))

    # split workload
    files_gpu0 = files[::2]
    files_gpu1 = files[1::2]

    mp.set_start_method("spawn", force=True)

    q0 = mp.Queue(maxsize=8)
    q1 = mp.Queue(maxsize=8)

    # =========================
    # START GPU PROCESSES
    # =========================
    p0 = mp.Process(target=gpu_worker, args=(0, q0, args))
    p1 = mp.Process(target=gpu_worker, args=(1, q1, args))

    p0.start()
    p1.start()

    # =========================
    # START PREFETCH THREADS
    # =========================
    t0 = threading.Thread(target=prefetch_worker, args=(files_gpu0, q0))
    t1 = threading.Thread(target=prefetch_worker, args=(files_gpu1, q1))

    t0.start()
    t1.start()

    # wait I/O
    t0.join()
    t1.join()

    # stop GPU workers
    q0.put(None)
    q1.put(None)

    # wait GPU finish
    p0.join()
    p1.join()

    print("[DONE] All processing completed")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main(args)
