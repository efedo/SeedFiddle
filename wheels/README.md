# Optional offline wheelhouse

Place Windows x86-64 wheels for all runtime requirements and their transitive
dependencies in this directory. When `seed_vision.py --offline` is used, pip is
called with `--no-index --find-links <this directory>` and cannot contact an
online package index.

The final validated lab release will freeze the exact wheel set and record file
hashes. Do not mix wheels built for different Python or processor architectures.
