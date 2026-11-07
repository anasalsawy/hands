"""The scholar's own subsystems: his Arabic, his corpus, his authority.

Nothing here talks to a model. Normalisation, crawling, extraction and the
inheritance/zakat calculators are deterministic on purpose — a fatwa that is
reproducible is a fatwa that can be checked, and a number from a model is not a
number. The persona layers above *use* this; they do not decide anything with it.
"""
from __future__ import annotations
