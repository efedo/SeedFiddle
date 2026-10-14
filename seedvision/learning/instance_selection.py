"""One authoritative instance preference shared by computation and export."""


def select_instances(*, unet=None, stardist=None, procedural=None,
                     enabled=None, foreground_available=True):
    for name, value in (('unet_instances', unet), ('stardist_instances', stardist),
                        ('procedural_instances', procedural)):
        if value is not None and (enabled is None or name in enabled):
            if name == 'procedural_instances' and not foreground_available:
                return name, None, 'Foreground evidence is missing. Apply Foreground references or select a supported species library.'
            return name, value, ''
    return None, None, 'No enabled instance analysis has completed. Run an instance node.'
