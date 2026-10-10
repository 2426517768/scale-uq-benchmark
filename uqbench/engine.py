import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import json_write
from .distributions import make_quantiles
from .models import (ScaleR, calibration_gap, evidential_loss, gaussian_nll, make_model)

LEARNED = ["GaussianTCN", "RandomNoise", "FeaturePGD", "MixUp", "AdvReg", "Evidential",
           "SCALE-R", "SCALE-noagg", "SCALE-noentropy", "SCALE-noadv", "SCALE-nosim", "SCALE-k8", "SCALE-k32"]
MAIN = ["GaussianTCN", "MCDropout", "DeepEnsembles", "Evidential", "RandomNoise", "FeaturePGD",
        "MixUp", "AdvReg", "SCALE-R", "Isotonic", "SplitConformal", "OnlineQuantile",
        "ZeroGaussian", "RollingEmpirical", "EWMAGaussian", "GARCH-Norm"]
ABLATIONS = [x for x in LEARNED if x.startswith("SCALE-") and x != "SCALE-R"]


def settings(cfg, method):
    sc = cfg["scale"].copy()
    if method == "SCALE-noagg":
        sc.update(aggregation=False, entropy_weight=0)
    if method == "SCALE-noentropy":
        sc["entropy_weight"] = 0
    if method == "SCALE-noadv":
        sc["adversarial"] = False
    if method == "SCALE-nosim":
        sc["similarity_weight"] = 0
    if method == "SCALE-k8":
        sc["k"] = 8
    if method == "SCALE-k32":
        sc["k"] = 32
    return sc


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_device(allow_cpu=False):
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Exercise a real kernel: CUDA availability alone is insufficient on new GPU architectures.
        x = torch.randn(2, 30, 60, device=device)
        c = nn.Conv1d(30, 64, 3).to(device)
        c(x).sum().backward()
        synchronize(device)
        return device
    if not allow_cpu:
        raise RuntimeError("No working CUDA GPU. Run SETUP_WINDOWS.bat / doctor first; "
                           "CPU execution requires the explicit --allow-cpu flag.")
    return torch.device("cpu")


def bank_indices(data, seed, size):
    return np.random.default_rng(seed + 771).choice(len(data.y), min(size, len(data.y)), replace=False)


def refresh_bank(model, data, indices, batch, device):
    if not isinstance(model, ScaleR):
        return
    was_training = model.training
    model.eval()
    hs = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            x = torch.from_numpy(data.windows(indices[start:start + batch])).to(device)
            hs.append(model.encoder(x))
    model.bank_h = torch.cat(hs).detach()
    model.train(was_training)


def attack(model, x, y, cfg, kind):
    t = cfg["training"]
    was_training = model.training
    model.eval()
    original = x.detach()
    adv = original.clone()
    for _ in range(t["pgd_steps"]):
        adv.requires_grad_(True)
        out = model(adv)
        objective = gaussian_nll(out, y) if kind == "FeaturePGD" else calibration_gap(out, y)
        if isinstance(model, ScaleR) and model.sc.get("similarity_weight", 0):
            penalty = (model.sc["similarity_threshold"] - model.similarity(adv, original)).clamp_min(0).mean()
            objective = objective - model.sc["similarity_weight"] * penalty
        grad = torch.autograd.grad(objective, adv, only_inputs=True)[0]
        adv = adv.detach() + (t["epsilon"] / max(t["pgd_steps"], 1)) * grad.sign()
        adv = torch.maximum(torch.minimum(adv, original + t["epsilon"]), original - t["epsilon"]).clamp(-8, 8)
    model.train(was_training)
    return adv.detach()


def validate(model, data, cfg, device):
    model.eval()
    values = []
    with torch.no_grad():
        batch = cfg["training"]["eval_batch_size"]
        for start in range(0, len(data.y), batch):
            ids = np.arange(start, min(start + batch, len(data.y)))
            x = torch.from_numpy(data.windows(ids)).to(device)
            y = torch.from_numpy(data.y[ids]).to(device)
            out = model(x)
            loss = evidential_loss(out, y, 0) if "alpha" in out else gaussian_nll(out, y)
            values.append((float(loss), len(ids)))
    return sum(v * n for v, n in values) / sum(n for _, n in values)


def save_checkpoint(path, payload):
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(cfg, method, seed, train_data, val_data, directory, device):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    best_path, last_path = directory / "best.pt", directory / "last.pt"
    sc = settings(cfg, method)
    init_seed = seed
    if method.startswith("Ensemble-"):
        init_seed = seed + 1000 + int(method.split("-")[-1])
    seed_all(init_seed)
    allocated_before_model = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    model = make_model(cfg, method, sc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                                 weight_decay=cfg["training"]["weight_decay"])
    rng = np.random.default_rng(init_seed + 123)
    support = bank_indices(train_data, seed, sc["bank_size"])
    if (directory / "done.json").exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["state"])
        refresh_bank(model, train_data, support, cfg["training"]["eval_batch_size"], device)
        return model
    start_epoch, best, wait, history = 0, float("inf"), 0, []
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["state"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, best, wait, history = ck["epoch"] + 1, ck["best"], ck["wait"], ck["history"]
        rng.bit_generator.state = json.loads(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in ck["cuda_rng"]])
        print(f"Resume {method} seed={seed} at epoch {start_epoch + 1}", flush=True)
    if method.startswith("Ensemble-"):
        bootstrap = np.random.default_rng(init_seed).integers(0, len(train_data.y), len(train_data.y))
    else:
        bootstrap = np.arange(len(train_data.y))
    t = cfg["training"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, t["epochs"]):
        if wait >= t["patience"]:
            break
        synchronize(device)
        epoch_clock = time.perf_counter()
        refresh_bank(model, train_data, support, t["eval_batch_size"], device)
        model.train()
        ids_all = bootstrap[np.random.default_rng(init_seed + epoch * 100003).permutation(len(bootstrap))]
        synchronize(device)
        clock = time.perf_counter()
        total, count = 0., 0
        for start in range(0, len(ids_all), t["batch_size"]):
            ids = ids_all[start:start + t["batch_size"]]
            x = torch.from_numpy(train_data.windows(ids)).to(device)
            y = torch.from_numpy(train_data.y[ids]).to(device)
            if method == "RandomNoise":
                x = (x + torch.randn_like(x) * t["noise_sd"]).clamp(-8, 8)
            if method == "MixUp":
                lam = float(rng.beta(t["mixup_alpha"], t["mixup_alpha"]))
                perm = torch.from_numpy(rng.permutation(len(ids))).to(device)
                x, y = lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
            out = model(x)
            loss = evidential_loss(out, y, t["evidential_reg"]) if method == "Evidential" else gaussian_nll(out, y)
            if isinstance(model, ScaleR):
                loss = loss + sc["kl_weight"] * model.kl() / len(train_data.y)
                loss = loss + sc["entropy_weight"] * out["entropy"].mean()
            do_attack = method in ("FeaturePGD", "AdvReg") or (isinstance(model, ScaleR) and sc.get("adversarial", True))
            if do_attack:
                adversarial = attack(model, x, y, cfg, method)
                adversarial_out = model(adversarial)
                if isinstance(model, ScaleR):
                    loss = loss + .5 * gaussian_nll(adversarial_out, y)
                else:
                    loss = .5 * loss + .5 * gaussian_nll(adversarial_out, y)
                if method != "FeaturePGD":
                    loss = loss + cfg["scale"]["calibration_weight"] * calibration_gap(adversarial_out, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {method}, seed {seed}, epoch {epoch}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            count += len(ids)
        synchronize(device)
        seconds = time.perf_counter() - clock
        refresh_bank(model, train_data, support, t["eval_batch_size"], device)
        vloss = validate(model, val_data, cfg, device)
        synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_clock
        if not np.isfinite(vloss):
            raise FloatingPointError("Nonfinite validation loss.")
        if vloss < best - t["min_delta"]:
            best, wait = vloss, 0
            save_checkpoint(best_path, {"state": model.state_dict(), "method": method, "seed": seed,
                                       "best_epoch": epoch, "validation_loss": vloss})
        else:
            wait += 1
        history.append({"epoch": epoch + 1, "train_loss": total / count, "val_loss": vloss,
                        "train_seconds": seconds, "epoch_compute_seconds": epoch_seconds})
        save_checkpoint(last_path, {"state": model.state_dict(), "optimizer": optimizer.state_dict(),
                                     "epoch": epoch, "best": best, "wait": wait, "history": history,
                                     "numpy_rng": json.dumps(rng.bit_generator.state),
                                     "torch_rng": torch.get_rng_state(), "cuda_rng":
                                     torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        print(f"{method} s{seed} e{epoch+1}: train={total/count:.4f} val={vloss:.4f} {seconds:.1f}s", flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["state"])
    refresh_bank(model, train_data, support, t["eval_batch_size"], device)
    json_write(directory / "done.json", {"method": method, "seed": seed, "epochs": len(history),
               "history": history, "parameters": sum(p.numel() for p in model.parameters()),
               "training_seconds": sum(h["train_seconds"] for h in history),
               "epoch_compute_seconds": sum(h["epoch_compute_seconds"] for h in history),
               "runtime_scope": "epoch compute includes source-bank refresh and validation; excludes checkpoint I/O and startup",
               "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
               "incremental_peak_cuda_allocated_mb": (torch.cuda.max_memory_allocated(device)-allocated_before_model)/2**20 if device.type == "cuda" else 0,
               "reconstruction": "SCALE-R uses new specified code" if isinstance(model, ScaleR) else "new implementation"})
    return model


def predict_normalized(model, data, cfg, device, stochastic=False, seed=0):
    model.eval()
    if stochastic:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    seed_all(seed + 9959)
    batches = []
    with torch.no_grad():
        batch = cfg["training"]["eval_batch_size"]
        for start in range(0, len(data.y), batch):
            ids = np.arange(start, min(start + batch, len(data.y)))
            x = torch.from_numpy(data.windows(ids)).to(device)
            if stochastic:
                samples = [model(x) for _ in range(cfg["training"]["mc_samples"])]
                mus = torch.stack([o["mean"] for o in samples])
                ale = torch.stack([o["ale"] for o in samples]).mean(0)
                epi = mus.var(0, unbiased=False)
                out = {"mean": mus.mean(0), "ale": ale, "epi": epi, "var": ale + epi}
            else:
                out = model(x)
            batches.append({k: v.detach().cpu().numpy() for k, v in out.items()})
    model.eval()
    values = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    if "alpha" in values:
        df = 2 * values["alpha"]
        tscale = np.sqrt(values["beta"] * (1 + values["v"]) / (values["v"] * values["alpha"]))
        q = make_quantiles(values["mean"], values["var"], df, tscale)
    else:
        q = make_quantiles(values["mean"], values["var"])
    return {"mean": values["mean"], "sd": np.sqrt(values["var"]),
            "epi_sd": np.sqrt(values["epi"]), "quantiles": q}


def ensemble_predict(models, data, cfg, device, seed):
    outs = [predict_normalized(m, data, cfg, device, seed=seed) for m in models]
    mus = np.stack([o["mean"] for o in outs])
    mean = mus.mean(0)
    epi = mus.var(0)
    total = np.mean([o["sd"] ** 2 for o in outs], axis=0) + epi
    return {"mean": mean, "sd": np.sqrt(total), "epi_sd": np.sqrt(epi),
            "quantiles": make_quantiles(mean, total)}
