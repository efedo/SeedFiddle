"""Exact reference-frame transport; raster resizing is never registration."""
from dataclasses import replace
import cv2
import numpy as np


def validated_transform(value):
    matrix = np.asarray(value,np.float64)
    if matrix.shape != (3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('Reference coordinate transform must be a finite invertible 3×3 matrix.')
    return matrix


def reproject_raster(value, old_transform, new_transform, shape):
    if value is None:
        return None
    mapping = validated_transform(new_transform) @ np.linalg.inv(validated_transform(old_transform))
    values = np.asarray(value)
    if values.shape == tuple(shape) and np.allclose(mapping,np.eye(3),rtol=0,atol=1e-10):
        return values
    source = values.astype(np.uint8) if values.dtype == bool else values
    result = cv2.warpPerspective(source,mapping,(shape[1],shape[0]),flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return result.astype(values.dtype,copy=False)


def reproject_annotations(annotations, old_transform, new_transform):
    mapping = validated_transform(new_transform) @ np.linalg.inv(validated_transform(old_transform))
    result = []
    for annotation in annotations:
        point = annotation.hilum_point
        direction = annotation.hilum_direction
        if point is not None:
            points = [point]
            if direction is not None:
                points.append((point[0]+direction[0],point[1]+direction[1]))
            transformed = cv2.perspectiveTransform(np.asarray([points],np.float64),mapping)[0]
            point = tuple(float(value) for value in transformed[0])
            if direction is not None:
                delta = transformed[1]-transformed[0]
                direction = tuple(float(value) for value in delta/max(np.linalg.norm(delta),1e-12))
        elif direction is not None:
            # A projective direction has no unique transport without its anchor.
            direction = None
        result.append(replace(annotation,hilum_point=point,hilum_direction=direction))
    return tuple(result)


def reproject_bundle(bundle, new_transform, shape):
    if bundle.source_to_corrected is None:
        raise ValueError('Legacy references have no coordinate transform. Review their alignment before binding them to the current calibration.')
    old = bundle.source_to_corrected
    fields = {name:reproject_raster(getattr(bundle,name),old,new_transform,shape) for name in
              ('background','foreground','other','physical_edge','non_edge','annotated_seeds')}
    annotations = reproject_annotations(bundle.seed_annotations,old,new_transform)
    mapping = validated_transform(new_transform) @ np.linalg.inv(validated_transform(old))
    reviewed = []
    for item in annotations:
        # A complete source outline can become partial when the new canvas clips it.
        if bundle.annotated_seeds is not None:
            y, x = np.nonzero(bundle.annotated_seeds == item.seed_id)
            if len(x):
                points = cv2.perspectiveTransform(np.stack((x,y),axis=1).astype(np.float64)[None],mapping)[0]
                clipped = (~np.isfinite(points).all(axis=1) | (points[:,0]<0) |
                    (points[:,0]>shape[1]-1) | (points[:,1]<0) | (points[:,1]>shape[0]-1)).any()
                if clipped:
                    item = replace(item,shape_reviewed=False,outline_visibility='image_cutoff')
        if item.hilum_point is not None and not (0 <= item.hilum_point[0] < shape[1] and 0 <= item.hilum_point[1] < shape[0]):
            item = replace(item,hilum_point=None,hilum_direction=None)
        reviewed.append(item)
    annotations = tuple(reviewed)
    retained = set() if fields['annotated_seeds'] is None else set(np.unique(fields['annotated_seeds']))
    return replace(bundle,shape=tuple(shape),source_to_corrected=validated_transform(new_transform),
        seed_annotations=tuple(item for item in annotations if item.seed_id in retained),**fields)
