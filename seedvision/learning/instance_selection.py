"""One authoritative instance preference shared by computation and export."""

# Instance nodes in order of preference.
INSTANCE_NODE_IDS = ('unet_instances', 'stardist_instances', 'cellpose_sam_instances',
                     'procedural_instances')


def select_instances(*, unet=None, stardist=None, procedural=None, cellpose=None,
                     enabled=None, foreground_available=True):
    for name, value in zip(INSTANCE_NODE_IDS, (unet, stardist, cellpose, procedural)):
        if value is not None and (enabled is None or name in enabled):
            if name == 'procedural_instances' and not foreground_available:
                return name, None, 'Foreground evidence is missing. Apply Foreground references or select a supported species library.'
            return name, value, ''
    return None, None, 'No enabled instance analysis has completed. Run an instance node.'
