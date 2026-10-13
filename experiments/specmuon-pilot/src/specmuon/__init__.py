"""specmuon: spectral-prior Muon pilot package (independent of V16)."""

from .operators import apply_operator, bulk_spec_muon, exact_polar, head_spec_muon, muon_power
from .spec_muon import SpecMuon

__all__ = [
    "apply_operator",
    "exact_polar",
    "head_spec_muon",
    "bulk_spec_muon",
    "muon_power",
    "SpecMuon",
]
