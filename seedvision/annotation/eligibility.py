"""Shared eligibility for consumers that require complete physical contours."""
import numpy as np


def eligible_instances(labels, annotations):
    if labels is None:
        return None
    identifiers = [item.seed_id for item in annotations if item.shape_reviewed
                   and item.outline_visibility == 'complete' and not item.shape_exclusion_reason]
    from seedvision.annotation.continuity import summarize_instance_continuity
    disconnected = set(summarize_instance_continuity(labels).disconnected_identifiers)
    result = np.where(np.isin(labels, [value for value in identifiers if value not in disconnected]), labels, 0).astype(np.uint16)
    result.flags.writeable = False
    return result
