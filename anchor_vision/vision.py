"""Anchor Vision — main API. Intention-driven image compression for AI."""
import cv2
import numpy as np
import hashlib
import os
import time
from typing import Optional, Dict, Any, List
from .detect import detect_all, DetectionResult, ROI
from .compress import compress_image, diff_images, crop_roi, encode_crop

# No TTL — images stay in memory as long as the process runs.


class AnchorVision:
    """
    Intention-driven image compression for AI vision.

    Core rules:
    1. Never send original images. Crop or ask.
    2. If existing info answers the intention, use text. Otherwise use image.
    3. No priority between user and model ROIs. Both are kept. Compress what nobody cares about.
    4. Asking is cheaper than guessing.
    5. Change matters more than current state. Similar images only send diffs.
    6. User says "forget this" → it's forgotten. No questions.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache: Dict[str, Dict] = {}  # image_id -> {image, detection, phash}
        self._previous: Optional[str] = None  # last image_id
        self._cache_dir = cache_dir or "/tmp/anchor_vision_cache"
        os.makedirs(self._cache_dir, exist_ok=True)

    def _image_id(self, image: np.ndarray) -> str:
        """Generate a stable ID for an image."""
        data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]
        return "av_" + hashlib.md5(data).hexdigest()[:12]

    def _phash(self, image: np.ndarray) -> str:
        """Perceptual hash for similarity comparison."""
        small = cv2.resize(image, (32, 32))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        bits = (gray > mean).flatten()
        return "".join("1" if b else "0" for b in bits)

    def _phash_similarity(self, h1: str, h2: str) -> float:
        """Compare two perceptual hashes. Returns 0-1 similarity."""
        if len(h1) != len(h2):
            return 0.0
        same = sum(a == b for a, b in zip(h1, h2))
        return same / len(h1)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load an image from path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return image

    def _cache_image(self, image: np.ndarray, detection: DetectionResult) -> str:
        """Cache image and detection results. Returns image_id."""
        image_id = self._image_id(image)
        self._cache[image_id] = {
            "image": image,
            "detection": detection,
            "phash": self._phash(image),
            "cached_at": time.time(),
        }
        self._previous = image_id
        return image_id


    # ── Public API ──

    def see(
        self,
        image_path: Optional[str] = None,
        image_id: Optional[str] = None,
        intention: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point. Process an image with optional intention.

        Args:
            image_path: Path to image file (first call)
            image_id: Cached image ID (follow-up calls)
            intention: Why are you looking? Free text.

        Returns:
            {image_id, text, crops, uncertain, suggestion}
        """
        # Load or retrieve from cache
        if image_id and image_id in self._cache:
            cached = self._cache[image_id]
            image = cached["image"]
            detection = cached["detection"]
        elif image_path:
            image = self._load_image(image_path)
            detection = detect_all(image)
            image_id = self._cache_image(image, detection)

            # Check if similar to previous image
            if self._previous and self._previous != image_id and self._previous in self._cache:
                prev = self._cache[self._previous]
                similarity = self._phash_similarity(
                    self._cache[image_id]["phash"], prev["phash"]
                )
                if similarity > 0.90:
                    return self._handle_similar(image, prev["image"], image_id, intention)
        else:
            return {"error": "Provide image_path or image_id"}

        # Parse intention into ROIs
        intention_rois = self._parse_intention(intention, image, detection) if intention else []

        # Deduplicate: if intention matched an existing detection ROI, don't double-count
        seen_bboxes = set()
        deduped_rois = []
        for roi in intention_rois + list(detection.rois):
            key = tuple(int(v) for v in roi.bbox)
            if key not in seen_bboxes:
                seen_bboxes.add(key)
                deduped_rois.append(roi)

        all_rois = deduped_rois
        if not all_rois:
            # No ROIs at all — return text, suggest asking
            from .compress import generate_description
            text = generate_description(image, detection)
            return {
                "image_id": image_id,
                "text": text,
                "crops": [],
                "uncertain": [],
                "suggestion": "What would you like me to look at?",
            }

        # Compress — pass deduped ROIs, not raw detection + intention
        from .detect import DetectionResult
        deduped_detection = DetectionResult(
            rois=all_rois,
            saliency_map=detection.saliency_map,
            image_shape=detection.image_shape,
        )
        result = compress_image(image, deduped_detection)
        result["image_id"] = image_id
        return result

    def glance(self, image_path: str) -> Dict[str, Any]:
        """
        Quick look — return tiny thumbnail + text description.
        For the two-step observe flow: glance first, then focus.
        """
        image = self._load_image(image_path)
        detection = detect_all(image)
        image_id = self._cache_image(image, detection)

        # 64x64 thumbnail
        thumb = cv2.resize(image, (64, 64))
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
        thumb_b64 = __import__("base64").b64encode(buf).decode("utf-8")

        from .compress import generate_description
        text = generate_description(image, detection)

        # List detected items
        items = list(set(r.label for r in detection.rois))

        return {
            "image_id": image_id,
            "thumbnail_b64": thumb_b64,
            "text": text,
            "detected_items": items,
        }

    def focus(self, image_id: str, region: Optional[str] = None, bbox: Optional[tuple] = None) -> Dict[str, Any]:
        """
        Focus on a specific region of a cached image.

        Args:
            image_id: Cached image ID from glance() or see()
            region: Natural language region ("top left", "the face", etc.)
            bbox: Explicit bounding box (x, y, w, h)
        """
        if image_id not in self._cache:
            return {"error": f"Image {image_id} not in cache"}

        cached = self._cache[image_id]
        image = cached["image"]
        detection = cached["detection"]

        if bbox:
            roi = ROI(label="focus", bbox=bbox, source="explicit")
        elif region:
            roi = self._region_to_roi(region, image, detection)
            if roi is None:
                return {
                    "image_id": image_id,
                    "text": f"Could not locate region: {region}",
                    "crops": [],
                }
        else:
            return {"error": "Provide region or bbox"}

        crop = crop_roi(image, roi)
        return {
            "image_id": image_id,
            "text": f"Focused on: {roi.label}",
            "crops": [{
                "label": roi.label,
                "image_b64": encode_crop(crop),
                "bbox": roi.bbox,
                "source": roi.source,
            }],
        }

    def forget(self, image_id: str) -> Dict[str, str]:
        """Remove a cached image and any associated data."""
        if image_id in self._cache:
            del self._cache[image_id]
            return {"status": "forgotten", "image_id": image_id}
        return {"status": "not_found", "image_id": image_id}

    def clear_cache(self):
        """Clear all cached images."""
        self._cache.clear()
        self._previous = None

    # ── Internal ──

    def _handle_similar(
        self, current: np.ndarray, previous: np.ndarray,
        image_id: str, intention: Optional[str]
    ) -> Dict[str, Any]:
        """Handle images similar to previous — only send diffs."""
        diff_result = diff_images(current, previous)

        if diff_result["change_ratio"] < 0.05:
            return {
                "image_id": image_id,
                "text": diff_result["text"],
                "crops": [],
                "uncertain": [],
            }

        # Crop changed regions
        crops = []
        for roi in diff_result["changed_rois"]:
            crop = crop_roi(current, roi)
            if crop.size > 0:
                crops.append({
                    "label": roi.label,
                    "image_b64": encode_crop(crop),
                    "bbox": roi.bbox,
                    "source": "diff",
                })

        return {
            "image_id": image_id,
            "text": diff_result["text"],
            "crops": crops,
            "uncertain": [],
            "diff_from": self._previous,
        }

    def _parse_intention(
        self, intention: str, image: np.ndarray, detection: DetectionResult
    ) -> List[ROI]:
        """Parse intention text into ROIs. Basic keyword matching for now."""
        intention_lower = intention.lower()
        rois = []

        # Face-related intentions
        face_keywords = ["face", "expression", "crying", "eyes", "makeup",
                         "脸", "表情", "哭", "眼", "妆", "状态"]
        if any(k in intention_lower for k in face_keywords):
            face_rois = [r for r in detection.rois if r.label == "face"]
            for r in face_rois:
                r.source = "intention"
                rois.append(r)

        # Text-related intentions
        text_keywords = ["text", "read", "writing", "label", "sign",
                         "字", "读", "写", "标签", "牌"]
        if any(k in intention_lower for k in text_keywords):
            text_rois = [r for r in detection.rois if r.label == "text"]
            for r in text_rois:
                r.source = "intention"
                rois.append(r)

        # If intention mentions specific position
        h, w = image.shape[:2]
        if "left" in intention_lower or "左" in intention_lower:
            rois.append(ROI("left_region", (0, 0, w // 2, h), 0.5, "intention"))
        if "right" in intention_lower or "右" in intention_lower:
            rois.append(ROI("right_region", (w // 2, 0, w // 2, h), 0.5, "intention"))

        return rois

    def _region_to_roi(
        self, region: str, image: np.ndarray, detection: DetectionResult
    ) -> Optional[ROI]:
        """Convert natural language region to ROI."""
        region_lower = region.lower()
        h, w = image.shape[:2]

        # Check detection results first
        for r in detection.rois:
            if r.label in region_lower:
                return ROI(label=region, bbox=r.bbox, source="focus")

        # Positional
        positions = {
            "top left": (0, 0, w // 2, h // 2),
            "top right": (w // 2, 0, w // 2, h // 2),
            "bottom left": (0, h // 2, w // 2, h // 2),
            "bottom right": (w // 2, h // 2, w // 2, h // 2),
            "center": (w // 4, h // 4, w // 2, h // 2),
            "top": (0, 0, w, h // 2),
            "bottom": (0, h // 2, w, h // 2),
            "left": (0, 0, w // 2, h),
            "right": (w // 2, 0, w // 2, h),
        }
        for name, bbox in positions.items():
            if name in region_lower:
                return ROI(label=region, bbox=bbox, source="focus")

        return None
