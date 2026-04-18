"""Detection layer — find faces, text regions, objects, saliency."""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ROI:
    label: str
    bbox: tuple  # (x, y, w, h)
    confidence: float = 1.0
    source: str = "auto"  # "auto", "intention", "memory"


@dataclass
class DetectionResult:
    rois: List[ROI] = field(default_factory=list)
    saliency_map: Optional[np.ndarray] = None
    image_shape: tuple = (0, 0)

    @property
    def roi_coverage(self) -> float:
        """Fraction of image covered by ROIs."""
        if not self.rois or self.image_shape[0] == 0:
            return 0.0
        total_pixels = self.image_shape[0] * self.image_shape[1]
        roi_pixels = sum(r.bbox[2] * r.bbox[3] for r in self.rois)
        return min(roi_pixels / total_pixels, 1.0)


# ── Face Detection ──

_face_cascade = None

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def detect_faces(image: np.ndarray) -> List[ROI]:
    """Detect faces using OpenCV DNN (more robust than Haar cascade)."""
    try:
        return _detect_faces_dnn(image)
    except Exception:
        return _detect_faces_haar(image)


def _detect_faces_dnn(image: np.ndarray) -> List[ROI]:
    """DNN-based face detection (Caffe model, ships with OpenCV)."""
    import os
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    prototxt = os.path.join(model_dir, "deploy.prototxt")
    caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt):
        # Fall back to Haar if DNN model not downloaded
        return _detect_faces_haar(image)

    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    rois = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            rois.append(ROI(label="face", bbox=(x1, y1, x2 - x1, y2 - y1),
                          confidence=float(confidence), source="auto"))
    return rois


def _detect_faces_haar(image: np.ndarray) -> List[ROI]:
    """Fallback: Haar cascade face detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    # More sensitive settings
    faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    return [ROI(label="face", bbox=tuple(f), source="auto") for f in faces]


# ── Text Region Detection ──

_paddle_ocr = None

def _get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False,
                                 det=True, rec=False, cls=False)
    return _paddle_ocr


def detect_text_regions(image: np.ndarray) -> List[ROI]:
    """Detect text regions. Uses PaddleOCR if available, MSER as fallback."""
    try:
        return _detect_text_paddle(image)
    except (ImportError, Exception):
        return _detect_text_mser(image)


def _detect_text_paddle(image: np.ndarray) -> List[ROI]:
    """Detect text regions using PaddleOCR's detection model."""
    ocr = _get_paddle_ocr()
    result = ocr.ocr(image, det=True, rec=False, cls=False)

    rois = []
    if not result or not result[0]:
        return rois

    for line in result[0]:
        # line is a list of 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        pts = np.array(line, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        rois.append(ROI(label="text", bbox=(x, y, w, h), confidence=0.9, source="auto"))

    return rois[:5]


def _detect_text_mser(image: np.ndarray) -> List[ROI]:
    """Fallback: detect text regions using MSER + heuristics."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    mser = cv2.MSER_create()
    mser.setDelta(5)
    mser.setMinArea(60)
    mser.setMaxArea(int(h_img * w_img * 0.01))
    mser.setMaxVariation(0.25)
    regions, _ = mser.detectRegions(gray)

    bboxes = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        aspect = w / max(h, 1)
        area = w * h
        img_area = h_img * w_img
        if area > img_area * 0.05:
            continue
        if w < 8 or h < 8:
            continue
        if aspect < 0.15 or aspect > 12:
            continue
        if h > h_img * 0.3:
            continue
        bboxes.append((x, y, w, h))

    if not bboxes:
        return []

    merged = _merge_boxes(bboxes, overlap_thresh=0.3)
    result = []
    img_area = h_img * w_img
    for b in merged:
        if b[2] * b[3] > img_area * 0.15:
            continue
        result.append(ROI(label="text", bbox=b, confidence=0.7, source="auto"))
    return result[:5]


def _merge_boxes(boxes, overlap_thresh=0.3):
    """Simple NMS-like box merging."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    merged = []
    used = set()
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if i in used:
            continue
        mx, my, mw, mh = x1, y1, x1 + w1, y1 + h1
        for j, (x2, y2, w2, h2) in enumerate(boxes[i+1:], i+1):
            if j in used:
                continue
            # Check overlap
            ox = max(0, min(mx + mw, x2 + w2) - max(mx, x2))
            oy = max(0, min(my + mh, y2 + h2) - max(my, y2))
            if ox * oy > overlap_thresh * min(w1 * h1, w2 * h2):
                mx = min(mx, x2)
                my = min(my, y2)
                mw = max(mx + mw, x2 + w2)
                mh = max(my + mh, y2 + h2)
                used.add(j)
        merged.append((mx, my, mw - mx, mh - my))
        used.add(i)
    return merged[:10]  # Cap at 10 text regions


# ── Saliency ──

def compute_saliency(image: np.ndarray) -> np.ndarray:
    """Compute saliency map using OpenCV's static saliency."""
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)
    if success:
        return (saliency_map * 255).astype(np.uint8)
    return np.zeros(image.shape[:2], dtype=np.uint8)


# ── Full Detection Pipeline ──

def detect_all(image: np.ndarray) -> DetectionResult:
    """Run all detectors on an image."""
    result = DetectionResult(image_shape=image.shape[:2])
    result.rois.extend(detect_faces(image))
    result.rois.extend(detect_text_regions(image))
    result.saliency_map = compute_saliency(image)
    return result
