"""Pillow resampling constants shared by Habitat, Codex and replay.

Pillow <9.1 uses Image.NEAREST/BILINEAR/LANCZOS rather than Image.Resampling.
Keep this small module independent of any simulator or image-processing code.
"""
from PIL import Image

RESAMPLING = getattr(Image, "Resampling", Image)
