"""PyTorch models. SCALE-R is a new specified reconstruction, not recovered code."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, cin, cout, dilation, dropout):
        super().__init__()
        self.padding = 2 * dilation
        self.a = nn.Conv1d(cin, cout, 3, padding=self.padding, dilation=dilation)
        self.b = nn.Conv1d(cout, cout, 3, padding=self.padding, dilation=dilation)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.drop(F.relu(self.a(x)[..., :-self.padding]))
        z = self.drop(F.relu(self.b(z)[..., :-self.padding]))
        return F.relu(z + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, features, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(CausalBlock(features, hidden, 1, dropout),
                                 CausalBlock(hidden, hidden, 2, dropout),
                                 CausalBlock(hidden, hidden, 4, dropout))

    def forward(self, x):
        return self.net(x.transpose(1, 2))[..., -1]


class GaussianTCN(nn.Module):
    def __init__(self, features, hidden, dropout):
        super().__init__()
        self.encoder = Encoder(features, hidden, dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        out = self.head(self.encoder(x))
        ale = F.softplus(out[:, 1]) + 1e-4
        return {"mean": out[:, 0], "ale": ale, "epi": torch.zeros_like(ale), "var": ale}


class EvidentialTCN(nn.Module):
    def __init__(self, features, hidden, dropout):
        super().__init__()
        self.encoder = Encoder(features, hidden, dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, x):
        raw = self.head(self.encoder(x))
        mu = raw[:, 0]
        v = F.softplus(raw[:, 1]) + 1e-3
        alpha = F.softplus(raw[:, 2]) + 1.01
        beta = F.softplus(raw[:, 3]) + 1e-3
        ale = beta / (alpha - 1)
        epi = ale / v
        return {"mean": mu, "ale": ale, "epi": epi, "var": ale + epi,
                "v": v, "alpha": alpha, "beta": beta}


class ScaleR(nn.Module):
    def __init__(self, features, hidden, dropout, sc):
        super().__init__()
        self.encoder = Encoder(features, hidden, dropout)
        self.weight_mu = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        self.weight_rho = nn.Parameter(torch.full((hidden,), -3.))
        self.bias_mu = nn.Parameter(torch.zeros(()))
        self.bias_rho = nn.Parameter(torch.tensor(-3.))
        self.ale_head = nn.Linear(hidden, 1)
        self.embedding = nn.Linear(hidden, 16)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.sc = sc.copy()
        self.register_buffer("bank_h", torch.empty(0, hidden), persistent=False)

    def posterior_variance(self, h):
        return (h.square() * F.softplus(self.weight_rho).square()).sum(-1) + F.softplus(self.bias_rho).square()

    def forward(self, x):
        h = self.encoder(x)
        mu = h @ self.weight_mu + self.bias_mu
        raw_epi = self.posterior_variance(h)
        ale = F.softplus(self.ale_head(h).squeeze(-1)) + 1e-4
        entropy = torch.zeros_like(raw_epi)
        epi = raw_epi
        if self.sc.get("aggregation", True) and len(self.bank_h):
            query = F.normalize(self.embedding(h), dim=-1)
            # Frozen source-bank representation, but the query kernel and temperature learn.
            ref = F.normalize(self.embedding(self.bank_h).detach(), dim=-1)
            d = torch.cdist(query, ref).square()
            dist, idx = torch.topk(d, min(self.sc["k"], len(ref)), dim=-1, largest=False)
            temp = self.log_temperature.exp().clamp(.05, 5.)
            weights = torch.softmax(-dist / temp, dim=-1)
            ref_var = self.posterior_variance(self.bank_h).detach()
            pooled = (weights * ref_var[idx]).sum(-1)
            blend = self.sc["blend_self"]
            epi = blend * raw_epi + (1 - blend) * pooled
            entropy = (weights * torch.log(weights.clamp_min(1e-9))).sum(-1)
        return {"mean": mu, "ale": ale, "epi": epi, "var": ale + epi, "entropy": entropy}

    def kl(self):
        def normal_kl(mu, rho):
            sd = F.softplus(rho)
            return .5 * (mu.square() + sd.square() - 1 - 2 * torch.log(sd)).sum()
        return normal_kl(self.weight_mu, self.weight_rho) + normal_kl(self.bias_mu, self.bias_rho)

    def similarity(self, a, b):
        ea = F.normalize(self.embedding(self.encoder(a)), dim=-1)
        eb = F.normalize(self.embedding(self.encoder(b)).detach(), dim=-1)
        return torch.exp(-(ea - eb).square().sum(-1) / self.log_temperature.exp().detach().clamp(.05, 5.))


def gaussian_nll(out, y):
    var = out["var"].clamp_min(1e-6)
    return (.5 * (torch.log(var) + (out["mean"] - y).square() / var)).mean()


def evidential_loss(out, y, reg):
    v, a, b, mu = out["v"], out["alpha"], out["beta"], out["mean"]
    term = 2 * b * (1 + v)
    nll = .5 * torch.log(math.pi / v) - a * torch.log(term)
    nll = nll + (a + .5) * torch.log(v * (y - mu).square() + term) + torch.lgamma(a) - torch.lgamma(a + .5)
    return (nll + reg * (y - mu).abs() * (2 * v + a)).mean()


def calibration_gap(out, y):
    # Compare quantities in the same standardized-return unit; E|Normal error| = sqrt(2/pi)*SD.
    return F.smooth_l1_loss(math.sqrt(2 / math.pi) * out["var"].clamp_min(1e-6).sqrt(), (y - out["mean"]).abs())


def make_model(cfg, method, sc=None):
    t = cfg["training"]
    features = cfg.get("feature_count", 30)
    if method.startswith("SCALE"):
        return ScaleR(features, t["hidden"], t["dropout"], sc or cfg["scale"])
    if method == "Evidential":
        return EvidentialTCN(features, t["hidden"], t["dropout"])
    return GaussianTCN(features, t["hidden"], t["dropout"])
