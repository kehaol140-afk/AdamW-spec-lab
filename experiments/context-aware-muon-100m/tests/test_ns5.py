import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_aware_muon.ns5 import adjust_lr, lam_anchor, ns5

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _cos(a, b):
    a, b = a.double(), b.double()
    return float((a * b).sum() / (a.norm() * b.norm()))


def test_homogeneity_positive_scale():
    torch.manual_seed(0)
    for shape in [(768, 2432), (2432, 768), (512, 512)]:
        x = torch.randn(*shape)
        c = 3.71
        a, b = ns5(x), ns5(x * c)
        assert _cos(a, b) > 0.9995, (shape, _cos(a, b))  # residual = bf16 cast of cX (~1e-4 typical)
        assert abs(float(a.norm() / b.norm()) - 1) < 1e-2


def test_output_flattens_spectrum():
    torch.manual_seed(1)
    x = torch.randn(256, 512)
    y = ns5(x).float()
    g = y @ y.T
    r = min(x.shape)
    # NS5 yields singular values in ~[0.5, 1.5]; gram diag within that band squared
    d = torch.diagonal(g)
    assert float(d.min()) > 0.2 and float(d.max()) < 2.3


def test_2d_only_and_coeff_count():
    x3 = torch.randn(2, 3, 4)
    try:
        ns5(x3)
        raised = False
    except ValueError:
        raised = True
    assert raised
    try:
        ns5(torch.randn(4, 4), ns_coefficients=(1.0, 2.0))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_adjust_lr_modes():
    import math

    tall = torch.Size((768, 2432))  # rows < cols -> ratio 1
    wide = torch.Size((2432, 768))  # rows > cols -> sqrt(2432/768)
    assert abs(adjust_lr(1.0, None, tall) - 1.0) < 1e-12
    assert abs(adjust_lr(1.0, "original", wide) - math.sqrt(2432 / 768)) < 1e-12
    assert abs(adjust_lr(1.0, "match_rms_adamw", wide) - 0.2 * math.sqrt(2432)) < 1e-12


def test_lam_anchor():
    # code-truth anchor: native nesterov input = (1-mu)*g + mu*buf (torch lerp)
    assert abs(lam_anchor(0.95) - 0.05) < 1e-12
    assert abs(lam_anchor(0.90) - 0.10) < 1e-12


def test_no_svd_in_project_sources():
    banned = ("linalg.svd", "torch.svd", ".svd(")
    hits = []
    for p in SRC_ROOT.rglob("*.py"):
        txt = p.read_text()
        for b in banned:
            if b in txt:
                hits.append((str(p), b))
    assert not hits, hits
