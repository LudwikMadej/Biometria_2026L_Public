import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Any
from ._convert_to_int import convert_to_int


def load_image_to_numpy(source: Union[str, Path, Any]) -> np.ndarray:
    """
    Loads an image from the specified path or a file-like object
    into a NumPy array.

    Args:
        source (Union[str, Path, Any]): Path to the image file, or a file-like object
                                        with a .read() method.

    Returns:
        np.ndarray: Image data in RGB or RGBA format.

    Raises:
        FileNotFoundError: If the file path does not exist.
        IOError: If the file is not a valid image or cannot be opened.
    """
    if hasattr(source, "read"):
        image_to_open = source
        error_name = getattr(source, "name", "Uploaded File")
    else:
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        image_to_open = file_path
        error_name = str(file_path)

    try:
        with Image.open(image_to_open) as img:
            # Normalize to RGB or RGBA
            if img.mode == "RGBA":
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")

            return convert_to_int(np.array(img))

    except Exception as e:
        raise IOError(f"Failed to load image: {error_name}. Details: {e}")
