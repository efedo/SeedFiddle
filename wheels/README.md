# Optional offline wheelhouse

Runtime versions are governed by `config/runtime_dependencies.json`; the current
baseline is Python 3.12 x64. A laboratory release wheelhouse has not yet been
frozen or validated on the final target hardware.

Place Windows x86-64 wheels for all runtime requirements and their transitive
dependencies in this directory. When `seed_vision.py --offline` is used, pip is
called with `--no-index --find-links <this directory>` and cannot contact an
online package index.

The final validated lab release will freeze the exact wheel set and record file
hashes. Do not mix wheels built for different Python or processor architectures.
