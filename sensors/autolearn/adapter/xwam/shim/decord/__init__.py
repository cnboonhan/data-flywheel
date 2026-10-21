"""A stand-in for decord's VideoReader (no Linux arm64 wheels), the one call X-WAM's data loader makes:
VideoReader(path).get_batch(frame_ids).asnumpy() -> [T, H, W, 3] uint8, decoded with PyAV (its wheels bundle FFmpeg).
On PYTHONPATH ahead of site-packages in train_xwam.sbatch; the videos it reads are lerobot_to_xwam.py's short H.264
clips, so decoding up to the last requested frame is cheap."""
import av
import numpy as np


class _Batch:
    def __init__(self, a):
        self._a = a

    def asnumpy(self):
        return self._a


class VideoReader:
    def __init__(self, path, *args, **kwargs):
        self._path = str(path)
        self._frames = None

    def _decode(self, upto=None):
        with av.open(self._path) as c:
            out = []
            for i, f in enumerate(c.decode(video=0)):
                out.append(f.to_ndarray(format="rgb24"))
                if upto is not None and i >= upto:
                    break
        return out

    def __len__(self):
        if self._frames is None:
            self._frames = self._decode()
        return len(self._frames)

    def get_batch(self, indices):
        idx = [int(i) for i in indices]
        frames = self._frames if self._frames is not None else self._decode(max(idx))
        if max(idx) >= len(frames):
            raise IndexError(f"frame {max(idx)} past the end of {self._path} ({len(frames)} frames)")
        return _Batch(np.stack([frames[i] for i in idx]))
