"""Prepare MNIST and uploaded handwriting for the three classifiers."""

import cv2
import numpy as np
from PIL import Image, ImageOps


def normalize_images(images):
    """MNIST images are black background, white digit, with 0-255 pixels."""
    return images.astype(np.float32) / 255.0


def for_svm(images):
    return normalize_images(images).reshape(len(images), 784)


def for_cnn(images):
    return normalize_images(images).reshape(len(images), 28, 28, 1)


def preprocess_custom_image(image):
    """Return a centered MNIST-like image and inputs for every model.

    Raises ValueError for blank images instead of attempting a prediction.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = ImageOps.exif_transpose(image)
    # Composite transparent uploads over white, as most drawn digits are dark.
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        raise ValueError("The image is empty. Upload an image containing one digit.")

    # Border pixels give a practical background estimate for both light and
    # dark backgrounds without assuming the overall image is mostly blank.
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background_is_light = np.median(border) >= 128
    foreground = cv2.bitwise_not(gray) if background_is_light else gray
    foreground = cv2.GaussianBlur(foreground, (3, 3), 0)
    _, binary = cv2.threshold(
        foreground, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # A nearly uniform image is blank; Otsu alone may mark compression noise.
    if int(foreground.max()) - int(foreground.min()) < 20:
        raise ValueError("No clear digit was found. Try a darker, thicker stroke.")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    minimum_area = max(3, int(gray.size * 0.0002))
    valid = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum_area]
    if not valid:
        raise ValueError("No clear digit was found. Try a darker, thicker stroke.")
    # Keep all meaningful components so disconnected strokes (for example, a
    # dotted digit) are not accidentally discarded.
    mask = np.isin(labels, valid).astype(np.uint8) * 255
    points = cv2.findNonZero(mask)
    x, y, width, height = cv2.boundingRect(points)
    crop = cv2.bitwise_and(foreground, foreground, mask=mask)[y:y + height, x:x + width]
    scale = 20.0 / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (new_width, new_height), interpolation=interpolation)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    left = (28 - new_width) // 2
    top = (28 - new_height) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    normalized = normalize_images(canvas)
    return {
        "image": canvas,
        "mlp": normalized.reshape(1, 28, 28),
        "svm": normalized.reshape(1, 784),
        "cnn": normalized.reshape(1, 28, 28, 1),
    }
