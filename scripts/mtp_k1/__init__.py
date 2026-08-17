"""Isolated MTP K=1 helpers. Gemma RMSNorm is mandatory."""

from .gemma_rms import gemma_rms_norm, standard_rms_norm

__all__ = ["gemma_rms_norm", "standard_rms_norm"]
