"""Compact OpenCV 4.11.0 rotating-calipers compatibility for card geometry.

Only bounded CPU contour metadata is accepted. Float32 operation boundaries,
last-equal-area selection and the returned angle are preserved; no angle is
normalized to the loaded OpenCV convention. Hull generation uses the recorded
OpenCV runtime and has pinned-reference tests. See docs/OPENCV_MATH_NOTICES.md.

Adapted from OpenCV 4.11.0 modules/imgproc/src/rotcalipers.cpp:
Copyright (C) 2000, Intel Corporation, all rights reserved.
RotatedRect::points in modules/core/src/types.cpp additionally carries:
Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
Copyright (C) 2009, Willow Garage Inc., all rights reserved.
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* The names of the copyright holders may not be used to endorse or promote products
  derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation, OpenCV Foundation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages (including, but
not limited to, procurement of substitute goods or services; loss of use, data,
or profits; or business interruption) however caused and on any theory of
liability, whether in contract, strict liability, or tort (including negligence
or otherwise) arising in any way out of the use of this software, even if advised
of the possibility of such damage.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

MAX_POINTS = 1_000_000
MAX_ABS_COORDINATE = 64_000_000
ROTCALIPERS_SOURCE_SHA256 = 'bdc631e069f14836ddd5c811d56b0d7e71e8cb7383b4f272e87576d9ea9f65a0'
TYPES_SOURCE_SHA256 = '9d49b99da4de14e20c53d3dceb730627fceadb6c107bf9428505972d07c21d72'


def box_points_411(rectangle):
    """Preserve source float32 corner arithmetic and reflected far corners."""
    center, dimensions, angle = rectangle
    f = np.float32
    cx, cy = map(f, center)
    width, height = map(f, dimensions)
    angle = float(f(angle)) * math.pi / 180.0
    a, b = f(math.sin(angle)) * f(.5), f(math.cos(angle)) * f(.5)
    points = np.empty((4, 2), dtype=np.float32)
    points[0] = (f(f(cx-f(a*height))-f(b*width)), f(f(cy+f(b*height))-f(a*width)))
    points[1] = (f(f(cx+f(a*height))-f(b*width)), f(f(cy-f(b*height))-f(a*width)))
    points[2] = f(f(2)*np.array((cx, cy), np.float32)) - points[0]
    points[3] = f(f(2)*np.array((cx, cy), np.float32)) - points[1]
    return points


def _no_guard():
    return False


def min_area_rect_411(points, *, guard=None):
    """Return the source float32 rectangle from int32/float32 compact points."""
    if guard is None:
        guard = _no_guard
    values = np.asarray(points)
    if values.ndim == 3 and values.shape[1:] == (1, 2):
        values = values[:, 0, :]
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) > MAX_POINTS
            or values.dtype.kind not in 'if' or not np.isfinite(values).all()
            or np.any(np.abs(values.astype(np.float64)) > MAX_ABS_COORDINATE)):
        raise ValueError('Invalid or excessive compact rectangle coordinates')
    guard()
    if values.dtype.kind == 'i':
        values = values.astype(np.int32)
    else:
        values = values.astype(np.float32)
    hull = cv2.convexHull(values, clockwise=False, returnPoints=True)
    guard()
    if hull is None:
        return ((0.0, 0.0), (0.0, 0.0), 0.0)
    hull = hull[:, 0, :].astype(np.float32)
    n, f = len(hull), np.float32
    guard()
    if n <= 2:
        center, width, height, angle = np.zeros(2, np.float32), f(0), f(0), f(0)
        if n == 1:
            center = hull[0]
        elif n == 2:
            center = f(hull[0] + hull[1]) * f(.5)
            dx, dy = map(float, hull[1] - hull[0])
            width = f(math.sqrt(dx*dx + dy*dy))
            angle = f(math.atan2(dy, dx))
    else:
        vectors = np.empty_like(hull)
        inverses = np.empty(n, np.float32)
        left = bottom = right = top = 0
        for i in range(n):
            if i % 1024 == 0:
                guard()
            if hull[i, 0] < hull[left, 0]: left = i
            if hull[i, 0] > hull[right, 0]: right = i
            if hull[i, 1] > hull[top, 1]: top = i
            if hull[i, 1] < hull[bottom, 1]: bottom = i
            vectors[i] = hull[(i + 1) % n] - hull[i]
            dx, dy = map(float, vectors[i])
            inverses[i] = f(1 / math.sqrt(dx*dx + dy*dy))
        ax, ay = map(float, vectors[-1])
        orientation = f(0)
        for i, vector in enumerate(vectors):
            if i % 1024 == 0: guard()
            bx, by = map(float, vector)
            cross = ax*by - ay*bx
            if cross != 0:
                orientation = f(1 if cross > 0 else -1)
                break
            ax, ay = bx, by
        if orientation == 0:
            raise ValueError('Degenerate convex hull')
        sequence = [bottom, right, top, left]
        minimum, chosen = f(np.finfo(np.float32).max), None
        for k in range(n):
            if k % 1024 == 0: guard()
            v0, v1, v2, v3 = (vectors[i] for i in sequence)
            rotated = (v0, (v1[1], -v1[0]), (-v2[0], -v2[1]), (-v3[1], v3[0]))
            main = 0
            for i in range(1, 4):
                vx, vy = rotated[i]
                ox, oy = rotated[main]
                if f(f(vy*ox) + f(-vx*oy)) < 0:
                    main = i
            index = sequence[main]
            lead_x, lead_y = f(vectors[index] * inverses[index])
            a, b = ((lead_x, lead_y), (lead_y, -lead_x),
                    (-lead_x, -lead_y), (-lead_y, lead_x))[main]
            sequence[main] = (sequence[main] + 1) % n
            dx, dy = hull[sequence[1]] - hull[sequence[3]]
            width = f(f(dx*a) + f(dy*b))
            dx, dy = hull[sequence[2]] - hull[sequence[0]]
            height = f(f(-dx*b) + f(dy*a))
            area = f(width*height)
            if area <= minimum:
                minimum = area
                chosen = (sequence[3], a, b, width, height, sequence[0])
        left, a1, b1, width, height, bottom = chosen
        a2, b2 = -b1, a1
        c1 = f(f(a1*hull[left, 0]) + f(hull[left, 1]*b1))
        c2 = f(f(a2*hull[bottom, 0]) + f(hull[bottom, 1]*b2))
        inverse_det = f(f(1) / f(f(a1*b2) - f(a2*b1)))
        origin = np.array((f(f(f(c1*b2) - f(c2*b1))*inverse_det),
                           f(f(f(a1*c2) - f(a2*c1))*inverse_det)), np.float32)
        edge1, edge2 = np.array((f(a1*width), f(b1*width))), np.array((f(a2*height), f(b2*height)))
        center = origin + f(edge1 + edge2) * f(.5)
        dx, dy = map(float, edge1)
        width = f(math.sqrt(dx*dx + dy*dy))
        dx, dy = map(float, edge2)
        height = f(math.sqrt(dx*dx + dy*dy))
        angle = f(math.atan2(float(edge1[1]), float(edge1[0])))
    angle = f(float(f(angle * f(180))) / math.pi)
    guard()
    return (tuple(map(float, center)), (float(width), float(height)), float(angle))
