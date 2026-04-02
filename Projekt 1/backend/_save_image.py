import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union


def save_numpy_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """
    Saves a NumPy image array to a PNG or JPG file.

    Determines the output format from the file extension (.png, .jpg, .jpeg).
    Supports grayscale (1 channel), RGB (3 channels), and RGBA (4 channels).

    Args:
        image (np.ndarray): Image data as NumPy array.
                            Shape must be:
                            (H, W)           - grayscale
                            (H, W, 3)        - RGB
                            (H, W, 4)        - RGBA
        path (Union[str, Path]): Output file path.

    Raises:
        ValueError: If format is unsupported or array shape is invalid.
        IOError: If saving fails.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix not in [".png", ".jpg", ".jpeg"]:
        raise ValueError("Unsupported file format. Use .png, .jpg or .jpeg")

    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    try:
        if image.ndim == 2:
            mode = "L"  # grayscale
        elif image.ndim == 3:
            if image.shape[2] == 3:
                mode = "RGB"
            elif image.shape[2] == 4:
                if suffix in [".jpg", ".jpeg"]:
                    raise ValueError("JPEG does not support RGBA (alpha channel).")
                mode = "RGBA"
            else:
                raise ValueError("Unsupported channel count.")
        else:
            raise ValueError("Invalid image shape.")

        img = Image.fromarray(image.astype(np.uint8), mode=mode)

        # JPEG cannot store transparency
        if mode == "RGBA" and suffix in [".jpg", ".jpeg"]:
            img = img.convert("RGB")

        img.save(file_path)
        print("Image saved successfully at", path)

    except Exception as e:
        raise IOError(f"Failed to save image: {file_path}. Details: {e}")
