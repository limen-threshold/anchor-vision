"""Compression layer — crop ROIs, blur the rest, output text + crops."""
import cv2
import numpy as np
import base64
import io
from typing import List, Optional, Dict, Any
from .detect import ROI, DetectionResult


def crop_roi(image: np.ndarray, roi: ROI, padding: float = 0.15) -> np.ndarray:
    """Crop an ROI from the image with padding."""
    h, w = image.shape[:2]
    x, y, rw, rh = roi.bbox
    pad_x = int(rw * padding)
    pad_y = int(rh * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + rw + pad_x)
    y2 = min(h, y + rh + pad_y)
    return image[y1:y2, x1:x2]


def encode_crop(crop: np.ndarray, quality: int = 85) -> str:
    """Encode a crop as base64 JPEG."""
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def generate_description(image: np.ndarray, detection: DetectionResult) -> str:
    """Generate a text description of the scene from detection results."""
    h, w = image.shape[:2]
    parts = []

    # Basic scene info
    brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if brightness < 60:
        parts.append("low light / dark")
    elif brightness > 180:
        parts.append("bright / well-lit")

    # Dominant color tone
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[:, :, 0])
    mean_sat = np.mean(hsv[:, :, 1])
    if mean_sat < 40:
        parts.append("mostly gray/neutral tones")
    elif mean_hue < 15 or mean_hue > 165:
        parts.append("warm tones (red/orange)")
    elif 35 < mean_hue < 85:
        parts.append("cool tones (green)")
    elif 85 < mean_hue < 135:
        parts.append("cool tones (blue)")

    # ROI summary
    faces = [r for r in detection.rois if r.label == "face"]
    texts = [r for r in detection.rois if r.label == "text"]
    if faces:
        parts.append(f"{len(faces)} face(s) detected")
        for i, f in enumerate(faces):
            cx = f.bbox[0] + f.bbox[2] // 2
            pos = "left" if cx < w // 3 else ("right" if cx > 2 * w // 3 else "center")
            parts.append(f"  face {i+1}: {pos} of frame")
    if texts:
        parts.append(f"{len(texts)} text region(s)")

    return "; ".join(parts) if parts else "no notable features detected"


def compress_image(
    image: np.ndarray,
    detection: DetectionResult,
    intention_rois: Optional[List[ROI]] = None,
    blur_strength: int = 31,
    bg_scale: float = 0.125,
) -> Dict[str, Any]:
    """
    Compress an image: keep ROIs sharp, blur everything else.

    Returns dict with:
      - text: scene description
      - crops: list of {label, image_b64, bbox}
      - uncertain: list of {label, bbox} for uncertain regions
    """
    all_rois = list(detection.rois)
    if intention_rois:
        all_rois.extend(intention_rois)

    # Generate text description
    text = generate_description(image, detection)

    # If no ROIs found, return text only
    if not all_rois:
        return {
            "text": text,
            "crops": [],
            "uncertain": [],
            "suggestion": "What would you like me to look at?",
        }

    # Crop each ROI
    crops = []
    for roi in all_rois:
        crop = crop_roi(image, roi)
        if crop.size == 0:
            continue
        crops.append({
            "label": roi.label,
            "image_b64": encode_crop(crop),
            "bbox": roi.bbox,
            "source": roi.source,
        })

    return {
        "text": text,
        "crops": crops,
        "uncertain": [],
    }


def diff_images(
    current: np.ndarray,
    previous: np.ndarray,
    threshold: float = 30.0,
    min_change_area: float = 0.01,
) -> Dict[str, Any]:
    """
    Compare two similar images and return only the changed regions.

    Returns:
      - change_ratio: fraction of image that changed
      - changed_rois: list of ROIs for changed regions
      - text: description of changes
    """
    # Resize to match if needed
    if current.shape != previous.shape:
        previous = cv2.resize(previous, (current.shape[1], current.shape[0]))

    # Compute absolute difference
    diff = cv2.absdiff(
        cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY),
    )

    # Threshold
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of changed regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_pixels = current.shape[0] * current.shape[1]
    changed_pixels = cv2.countNonZero(mask)
    change_ratio = changed_pixels / total_pixels

    changed_rois = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area / total_pixels < min_change_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        changed_rois.append(ROI(
            label="changed_region",
            bbox=(x, y, w, h),
            confidence=min(area / total_pixels * 10, 1.0),
            source="diff",
        ))

    if change_ratio < 0.05:
        text = "Almost identical to previous — no notable changes"
    elif change_ratio < 0.30:
        text = f"Minor changes detected ({change_ratio:.0%} of image)"
    else:
        text = f"Significant changes ({change_ratio:.0%} of image)"

    return {
        "change_ratio": change_ratio,
        "changed_rois": changed_rois,
        "text": text,
    }
