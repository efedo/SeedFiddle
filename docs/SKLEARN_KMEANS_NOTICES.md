# scikit-learn MiniBatchKMeans notice

The CUDA unit-weight dense centre update in
`seedvision/cuda/plantcv_kmeans_minibatch.py` adapts the ordered update logic in
scikit-learn 1.9.1, `sklearn/cluster/_k_means_minibatch.pyx`.
The pinned source SHA-256 is
`b6ffd63a5a9c6f9162d77e1568911fd3540c4ceaaa8d5243cb341470e758fff5`.
This implementation uses CUDA threads for independent centre-feature sums while
preserving source row order, unit-weight count updates and empty-cluster copies.
It does not include the complete estimator's initialization, batch sampling,
reassignment or convergence loop.

The RNG call adapter in `seedvision/cuda/plantcv_kmeans_random.py` and the
byte-valued CUDA initializer in `seedvision/cuda/plantcv_kmeans_plusplus.py` follow
the unit-weight call order and k-means++ selection algorithm in scikit-learn1.9.1
`sklearn/cluster/_kmeans.py`, SHA-256
`0cf335b8b9b4abe67259ccdcbaf07f02bd7feee31e63a06a1cce32a87fcd375d`.
Squared byte distances, cumulative prefixes and potentials are exact integers
within the initializer's preflight domain. The bounded complete byte-feature
estimator is implemented in `seedvision/cuda/plantcv_kmeans_fit.py`.

The compact controller/reassignment planner in
`seedvision/cuda/plantcv_kmeans_control.py` adapts `_random_reassign`,
`_mini_batch_convergence` and the low-count reassignment branch of
`_mini_batch_step` from the same pinned `_kmeans.py` source. Dense updates and
centre writes remain CUDA; the helper handles bounded count snapshots, RNG row
selection and scalar control with certified inertia inputs. The complete bounded
fitting loop composes these helpers and certified source reduction intervals.

`seedvision/cuda/plantcv_kmeans_training_score.py` adapts the source Lloyd
centre-norm-minus-dot label score and grouped-four direct row distances from
`_k_means_lloyd.pyx` and `_k_means_common.pyx`. Their pinned SHA-256 values are
`66ba7cad80c8000e350c0dd4014e790b424e7c0157b479c98fc7610071aa2a92`
and `9a05c0184b411fd1d68c9033e265d4b41824f01f1a66d6c494cfb11f1780b489`.
Byte/integral totals are exact within declared limits; fractional winners are
guarded and source aggregate values are bounded by compact certified intervals.
`seedvision/cuda/plantcv_kmeans_intervals.py` preserves convergence expression/
branch order with interval decisions. It does not claim arbitrary threaded
fractional byte equality.

`seedvision/cuda/plantcv_kmeans_fit.py` follows the same pinned `_kmeans.py`
parameter resolution, validation/subset selection, strict initialization choice,
RNG/update/reassignment/convergence order and final labels/inertia. Its scope is
the bounded byte-valued k-means++ unit-weight estimator path; image/directory
ingestion and native training workflow integration remain separate.
`seedvision/cuda/plantcv_kmeans_tolerance.py` bounds the source public-tolerance
scaling and NumPy two-pass variance/centre-shift expressions using exact integer
moments and rational outward error bounds. NumPy2.2.6 `numpy/_core/_methods.py`
is pinned at SHA-256
`41068be3404b05b582867995683f73619b41a9fc72b9f0cc7983f55f53131146`.
The variance expression is independently implemented; no NumPy source is copied.

The following notice is retained from the pinned distribution's `COPYING` file.

BSD 3-Clause License

Copyright (c) 2007-2026 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
