> Start here: read [HANDOFF.md](HANDOFF.md) before inspecting or changing the project.

# Agent instructions

- Preserve the single `seed_vision.py` launch point and native PySide6 desktop architecture. Do not introduce a server, installer, or bundled executable.
- Keep full-resolution analysis tensors and reusable intermediates on the PyTorch CUDA device unless a CPU transfer is essential for Qt display or compact metadata.
- Treat `images/` as committed test fixtures. Do not add generated output from `artifacts/`.
- Preserve node-local caching and recompute only a changed node and its dependents.
- Run the focused tests for each change and the full `unittest` suite before hand-off.
- Keep exposed node controls honest: every parameter must affect a calculation, not merely validation or presentation.
