# Naive Bayes Gaussian training attribution

`seedvision/cuda/plantcv_naive_bayes_training.py` adapts the pairwise summation
structure from NumPy 2.2.6 `numpy/_core/src/umath/loops_utils.h.src`, its default
8,192-element outer reduction iteration and ordered accumulation, and the
Gaussian operation order from SciPy 1.15.3 `scipy/stats/_kde.py` and `_stats.pyx`.
The pinned hashes are recorded in that module. Source references:

- https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_utils.h.src
- https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/reduction.c
- https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/ufunc_object.c
- https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_arithm_fp.dispatch.c.src
- https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/multiarray/nditer_constr.c
- https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/include/numpy/ndarraytypes.h
- https://github.com/scipy/scipy/blob/v1.15.3/scipy/stats/_kde.py
- https://github.com/scipy/scipy/blob/v1.15.3/scipy/stats/_stats.pyx

New channel records identify `numpy-2.2.6-buffer8192-v1`. This is a fixed pinned
reference profile: changing the native process's NumPy buffer setting does not
change the CUDA algorithm. Reference runs with a customized buffer setting are
outside this profile. The previous unbuffered profile remains readable without
rewriting historical records. The extra source hashes are calculation constants
captured by the existing loaded-module provenance; the original source-pin
record and model TSV format remain unchanged.

The additive `numpy-2.2.6-buffer8192-constant-v1` profile preserves constant-byte
source singularity when the weighted mean equals that byte. Otherwise, positive
roundoff covariance retains the actual CUDA ordered residual products and sums.
For `u=2^-53` and `gamma=(n+32)u/(1-(n+32)u)`, its covariance allowance is
`4*gamma*covariance`; it does not add a variance floor or the varying-column
mean-uncertainty term. The guarded covariance upper bound below `2^-12`, Scott
factor at most one and whitening interval put other byte-bin exponents below
`-2048`. The peak has a separate multiplicative error allowance; normalization
still includes explicitly empirical runtime allowances, not a vendor-independent
certificate. Full sample/PDF reductions stay CUDA. The stdlib record codec
audits only compact uniform-term scalar facts using memoized pairwise subtree
lengths and the actual outer addition order, without constructing a sample array.
It requires the exact source-profile mean, validates the positive covariance
interval and budgets, and retains the constant byte value. Means slightly above
255 are accepted only when this exact audit reproduces them. Historical record
shapes stay unchanged; mixed varying/constant channels belong to the new profile
family. These numeric checks do not establish record authenticity or scientific
validity.

The input/output behavior follows PlantCV 4.11.3 `plantcv/learn/naive_bayes.py`
(SHA-256 `f59c95c493d189b6110d47a60a97c86e306ab87583a2f128eff8c389aa6b56eb`),
whose repository declares MPL-2.0. The new parser implements its behavior;
PlantCV's Python source is not copied into the native runtime.
https://github.com/danforthcenter/plantcv/blob/77e3322fd0cd5a9bef18bb99402f20c8f2db87a9/plantcv/learn/naive_bayes.py

## NumPy notice

Copyright (c) 2005-2024, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided
  with the distribution.
* Neither the name of the NumPy Developers nor the names of any
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## SciPy notice

Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
