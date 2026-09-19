"""Bounded evaluation storage: one decoded image at a time, disk-backed search."""
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil


class PredictionSpool:
    def __init__(self, maximum_bytes=8*1024**3):
        self._temporary = TemporaryDirectory(prefix='seedfiddle-evaluation-')
        self.root = Path(self._temporary.name)
        self.maximum_bytes = int(maximum_bytes)
        self.bytes = 0
        self.count = 0

    def append(self, value):
        import torch
        def byte_size(item):
            if torch.is_tensor(item):
                return item.numel()*item.element_size()
            return sum(byte_size(v) for v in (item.values() if isinstance(item,dict) else item))
        estimate = byte_size(value) + 65536
        if self.bytes+estimate > self.maximum_bytes or shutil.disk_usage(self.root).free < estimate+64*1024**2:
            raise MemoryError('Evaluation spool budget or free disk space exhausted. Evaluate a smaller frozen shard or increase maximum_prediction_bytes.')
        destination = self.root/f'{self.count}.pt'
        torch.save(value,destination)
        self.bytes += destination.stat().st_size
        self.count += 1

    def __len__(self):
        return self.count

    def __iter__(self):
        import torch
        for index in range(self.count):
            yield torch.load(self.root/f'{index}.pt',map_location='cpu',weights_only=True)

    def close(self):
        self._temporary.cleanup()


class LabelSequence:
    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        from seedvision.learning.data import read_label_image
        for path in self.paths:
            yield read_label_image(path)


def append_thumbnail(panels, panel):
    import cv2
    if len(panels) < 12:
        scale = min(1.,900/panel.shape[1])
        panels.append(cv2.resize(panel,(round(panel.shape[1]*scale),max(1,round(panel.shape[0]*scale)))))
