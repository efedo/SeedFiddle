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


def connected_instances(labels, annotations):
    """Keep every connected seed not explicitly excluded, reviewed or not.

    Training and reference prototypes use these: an unlabelled seed would train as
    background, and reviewed seeds alone can sit in one part of the image.
    """
    if labels is None:
        return None
    from seedvision.annotation.continuity import summarize_instance_continuity
    blocked = {item.seed_id for item in annotations if item.shape_exclusion_reason}
    blocked.update(summarize_instance_continuity(labels).disconnected_identifiers)
    result = np.where(np.isin(labels, list(blocked)), 0, labels).astype(np.uint16)
    result.flags.writeable = False
    return result
