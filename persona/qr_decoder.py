
from typing import Dict, Any, Optional, Tuple, Union
import re
import numpy as np
import cv2  # pip install opencv-python

ImageInput = Union[str, bytes, "np.ndarray"]
VCARD_BEGIN, VCARD_END = "BEGIN:VCARD", "END:VCARD"

class QRDecoder:
    """Decode QR codes from path/bytes/np.ndarray and classify as URL or vCard."""
    def __init__(self, prefer: tuple[str, ...] = ("url", "vcard")) -> None:
        self._prefer = prefer
        self._detector = cv2.QRCodeDetector()

    # ---------- public API ----------
    def decode(self, image: ImageInput) -> Dict[str, Any]:
        """Return {"url": "..."} or {"vcard": {...}} or {}."""
        img = self._load_image(image)
        if img is None:
            return {}
        candidates = self._decode_multi(img) or self._decode_single(img)
        for payload in candidates:
            kind, data = self._classify_payload(payload)
            if kind in self._prefer:
                return data
        return {}

    # ---------- helpers ----------
    def _load_image(self, image: ImageInput) -> Optional[np.ndarray]:
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, bytes):
            arr = np.frombuffer(image, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if isinstance(image, str):
            return cv2.imread(image, cv2.IMREAD_COLOR)
        return None

    def _decode_multi(self, img: np.ndarray) -> list[str]:
        try:
            texts, points, _ = self._detector.detectAndDecodeMulti(img)
            if points is not None and texts:
                return [t for t in texts if t]
        except Exception:
            pass
        return []

    def _decode_single(self, img: np.ndarray) -> list[str]:
        text, pts, _ = self._detector.detectAndDecode(img)
        return [text] if (pts is not None and text) else []

    def _classify_payload(self, payload: str) -> Tuple[str, Dict[str, Any]]:
        s = payload.strip()
        if re.match(r"^https?://", s, re.IGNORECASE):
            return "url", {"url": s}
        U = s.upper()
        if U.startswith(VCARD_BEGIN) and VCARD_END in U:
            return "vcard", {"vcard": self._parse_vcard(s)}
        return "text", {"text": s}

    def _parse_vcard(self, vcard_text: str) -> Dict[str, Any]:
        text = vcard_text.replace("\r\n", "\n").replace("\r", "\n")
        # Unfold continuations
        unfolded = []
        for line in text.split("\n"):
            if line.startswith((" ", "\t")) and unfolded:
                unfolded[-1] += line.lstrip()
            else:
                unfolded.append(line)

        fields: Dict[str, Any] = {}
        for line in unfolded:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.upper().split(";", 1)[0]
            val = val.strip()

            if key == "FN":
                fields["name"] = val
            elif key == "N":
                parts = val.split(";")
                given = parts[1] if len(parts) > 1 else ""
                family = parts[0] if len(parts) > 0 else ""
                fields.setdefault("name", " ".join(p for p in [given, family] if p).strip())
            elif key == "TITLE":
                fields["title"] = val
            elif key == "ORG":
                fields["company"] = val
            elif key == "ADR":
                parts = val.split(";")
                city = parts[3] if len(parts) > 3 else ""
                region = parts[4] if len(parts) > 4 else ""
                country = parts[6] if len(parts) > 6 else ""
                loc = ", ".join(x for x in [city, region or country] if x)
                if loc:
                    fields["location"] = loc
            elif key == "EMAIL":
                fields.setdefault("emails", []).append(val)
            elif key == "TEL":
                fields.setdefault("phones", []).append(val)
            elif key == "URL":
                fields.setdefault("urls", []).append(val)
        return fields
