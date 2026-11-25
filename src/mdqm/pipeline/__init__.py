"""The bootstrap pipeline: probe -> deterministic stages (D1..D10) -> agentic stages (A1..A2).

Each stage is a module under :mod:`mdqm.pipeline.stages` with one job, its own input model and a
report that states what it found, what it skipped and what it could not read. The stages are the
code half of DESIGN.md section 10: they compute, they never approve.
"""
