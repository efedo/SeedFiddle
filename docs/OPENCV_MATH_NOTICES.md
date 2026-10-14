# OpenCV mathematical-port notices

The compact card rectangle compatibility path in
[plantcv_minrect.py](../seedvision/cuda/plantcv_minrect.py) follows OpenCV4.11.0
[rotcalipers.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/rotcalipers.cpp)
and `RotatedRect::points` in
[types.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/types.cpp).
It retains float32 operation boundaries, equal-area tie order and source corner
arithmetic. Only bounded contour metadata is processed on CPU; no dense raster
transfers for these calculations. Copyright/license conditions are retained in
the module and the notices below. The types source additionally carries:

```text
Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
Copyright (C) 2009, Willow Garage Inc., all rights reserved.
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
```

The automatic-card preprocessing in
[plantcv_card_preprocess.py](../seedvision/cuda/plantcv_card_preprocess.py) follows
OpenCV 4.11.0's byte Gaussian fixed-point kernel and rounding in
[smooth.dispatch.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/smooth.dispatch.cpp)
and [smooth.simd.hpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/smooth.simd.hpp).
Compact float32 kernel digests are generated from the pinned 4.11.0.86 wheel.
Full rasters and separable filter intermediates remain on CUDA.

The shared adaptive-mean decision certificate in
[plantcv_thresholds.py](../seedvision/cuda/plantcv_thresholds.py) follows the scalar
double, SIMD float32 and SHIFT23 integer branches in
[box_filter.simd.hpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/box_filter.simd.hpp),
and singleton-axis dispatch in
[box_filter.dispatch.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/box_filter.dispatch.cpp).
It withholds binary decisions when the supported arithmetic branches disagree.
These sources carry the license agreement retained below; the box-filter source
additionally retains this copyright line:

```text
Copyright (C) 2000-2008, 2018, Intel Corporation, all rights reserved.
```

The compact ellipse compatibility path in
[plantcv_size.py](../seedvision/cuda/plantcv_size.py) follows the deterministic
ill-conditioned `fitEllipseNoDirect` branch in OpenCV 4.11.0
[shapedescr.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/shapedescr.cpp).
Only bounded contour metadata is processed on CPU. The source file carries the
same Intel notice retained below; no image raster transfers for this calculation.

The filled-circle scan in
[plantcv_color_card.py](../seedvision/cuda/plantcv_color_card.py) follows OpenCV
4.11.0 [drawing.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/drawing.cpp)
integer midpoint updates and horizontal spans. Only compact span metadata is
constructed on CPU; the labelled mask is written on CUDA. The drawing source's
notice is retained here:

```text
Intel License Agreement
For Open Source Computer Vision Library

Copyright (C) 2000, Intel Corporation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* The name of Intel Corporation may not be used to endorse or promote products
  derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
```

The Gaussian kernel constants in
[plantcv_extended_thresholds.py](../seedvision/cuda/plantcv_extended_thresholds.py)
follow the dyadic 3/5/7/9-tap tables in OpenCV 4.11.0
[smooth.dispatch.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/smooth.dispatch.cpp).
Adaptive threshold rounding/polarity follow
[thresh.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/thresh.cpp).
The CUDA implementation computes exact integer weighted sums in bounded tiles;
it does not vendor OpenCV's filtering implementation. The source-file notice is
retained below alongside the earlier conversion notices.

```text
License Agreement
For Open Source Computer Vision Library

Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
Copyright (C) 2009, Willow Garage Inc., all rights reserved.
Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistribution's of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistribution's in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* The name of the copyright holders may not be used to endorse or promote products
  derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
```

The byte LAB conversion and compact constant tables in
[plantcv_range.py](../seedvision/cuda/plantcv_range.py) follow OpenCV 4.11.0
[color_lab.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/color_lab.cpp),
[color.simd_helpers.hpp](https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/color.simd_helpers.hpp)
and the `f32_cbrt` rational approximation in
[softfloat.cpp](https://github.com/opencv/opencv/blob/4.11.0/modules/core/src/softfloat.cpp).
OpenCV is distributed under its
[Apache License 2.0](https://github.com/opencv/opencv/blob/4.11.0/LICENSE).
The approximation is attributed there to Ken Turkowski. No OpenCV binary or
complete SoftFloat/fdlibm implementation is vendored by this mathematical port.

The following notices are retained from the transitive source file. They supplement
the project's MIT license and OpenCV's license; they are not scientific validation.

```text
This C source file is part of the SoftFloat IEEE Floating-Point Arithmetic
Package, Release 3c, by John R. Hauser.
Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017 The Regents of the
University of California. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions, and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

FDLIBM licenses:

Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.

Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.
```
