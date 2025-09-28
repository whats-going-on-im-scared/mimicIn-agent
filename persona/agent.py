# agent.py
from __future__ import annotations

import base64
import io
import os
import re
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from google import genai
import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2  # OpenCV QR decoder

from google.adk.agents import Agent

# ============================================================
# =============  (moved from profile_tools.py)  ==============
# ============================================================

# ------------------ LinkedIn URL validation ------------------

_LINKEDIN_PROFILE_RE = re.compile(
    r"""^https?://
        (?:[a-z]{2,3}\.)?            # optional subdomain like 'www.'
        linkedin\.com/
        (?:
            (?:mwlite/)?in/          # /in/ or /mwlite/in/
            [^/?#]+                  # profile slug
          |
            profile/view\?id=\d+     # legacy numeric profile pattern
        )
        (?:[/?#].*)?                 # optional extra
        $""",
    re.IGNORECASE | re.VERBOSE,
)


def _is_linkedin_profile_url(url: str) -> bool:
    return bool(_LINKEDIN_PROFILE_RE.match((url or "").strip()))


# ------------------ Image helpers ------------------

def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def _decode_qr_cv2(image_bytes: bytes) -> Optional[str]:
    """Return first decoded QR string or None."""
    img = _bytes_to_bgr(image_bytes)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    if points is not None and isinstance(data, str) and data.strip():
        return data.strip()

    # retry with upscale (helps tiny QRs)
    h, w = img.shape[:2]
    if min(h, w) < 600:
        scale = max(1.5, 600.0 / min(h, w))
        img2 = cv2.resize(img, (int(w * scale), int(h * scale)))
        data2, pts2, _ = detector.detectAndDecode(img2)
        if pts2 is not None and isinstance(data2, str) and data2.strip():
            return data2.strip()
    return None


# ------------------ Tools ------------------

def qr_to_vcard_or_url(image_bytes: bytes, mime_type: str | None = None) -> dict:
    try:
        decoded = _decode_qr_cv2(image_bytes)
    except UnidentifiedImageError:
        return {"status": "error", "message": "Invalid image data: not a recognizable image."}
    except Exception:
        return {"status": "error", "message": "Unable to decode QR from the provided image."}

    if not decoded:
        return {"status": "error", "message": "No QR code found in image."}

    text = decoded.strip()

    # Normalize missing scheme like "linkedin.com/in/slug"
    lower = text.lower()
    if lower.startswith("linkedin.com/") or lower.startswith("www.linkedin.com/"):
        text = "https://" + text.lstrip()
        lower = text.lower()

    # Strict LinkedIn profile URL validation
    if not _is_linkedin_profile_url(text):
        return {"status": "error", "message": "We only accept LinkedIn profile URLs at the moment."}

    return {"status": "success", "linkedin_url": text}


def render_prompt_from_json(profile: dict, extra_context: str | None = None) -> dict:
    """
    Turn profile JSON into persona instructions for a live conversation agent.
    """
    name = profile.get("name") or "Recruiter"
    position = profile.get("position") or "Recruiter"
    company = profile.get("company") or "Company"
    location = profile.get("location") or "United States"

    instructions = f"""
    You are {name}, {position} at {company}, located in {location}.
    You are interviewing a candidate at a college job fair who has expressed interest in joining the firm.
    Please ask questions {name} would be likely to ask given their background.

    Rules:
    - Focus on early-career talent (this is a college fair).
    -Generate replies based off of your position. For example, if you are software engineering, offer insights about a framework; If you are a product manager, for example, give insights about agile principles
    - Keep replies short and professional (target ~3 minutes).
    - No markdown/emoji, use plain English.
    - If you're unsure, ground yourself first (google_search tool).
    - End with "\\END_CONVERSATION\\" and feedback on the candidate's performance.
    {extra_context or ""}
    """.strip()

    return {
        "status": "success",
        "persona_instructions": instructions,
        "profile": {
            "name": name,
            "position": position,
            "company": company,
            "location": location,
            "url": profile.get("url"),
        },
    }


# ------------------ LinkedIn scraper wrapper ------------------

def linkedin_profile_extractor(
        linkedin_url: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        headless: bool = True,
        chromedriver: Optional[str] = None,
) -> dict:
    """
    Wraps linkedin_scrape_chrome.py (Selenium). Returns:
      { "name": ..., "position": ..., "company": ..., "location": ..., "url": ... }
      or { "error": "..." }
    """
    # Normalize missing scheme like "linkedin.com/in/slug" before validation
    normalized_url = linkedin_url.strip()
    lower = normalized_url.lower()
    if lower.startswith("linkedin.com/") or lower.startswith("www.linkedin.com/"):
        normalized_url = "https://" + normalized_url

    if not _is_linkedin_profile_url(normalized_url):
        return {"error": "Invalid LinkedIn profile URL."}

    # Resolve script path (repo local first, then /mnt/data)
    script = (Path(__file__).parent / "linkedin_scrape_chrome.py")
    if not script.exists():
        alt = Path("/mnt/data/linkedin_scrape_chrome.py")
        script = alt if alt.exists() else script
    if not script.exists():
        return {"error": "linkedin_scrape_chrome.py not found"}

    email = email or os.getenv("EMAIL")
    password = password or os.getenv("PASSWORD")
    if not email or not password:
        return {"error": "Missing EMAIL/PASSWORD for LinkedIn"}

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        out_path = f.name

    cmd: List[str] = [
        sys.executable, str(script), normalized_url,  # Use normalized_url
        "--email", email, "--password", password,
        "--output", out_path,
    ]
    if headless:
        cmd.append("--headless")
    if chromedriver:
        cmd.extend(["--chromedriver", chromedriver])

    try:
        run = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Read file written by the scraper
        with open(out_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            # Update the URL in the returned data to the normalized version
            data["url"] = normalized_url
            return data
    except subprocess.CalledProcessError as e:
        return {"error": f"scraper failed: {e.stderr or e.stdout or str(e)}"}
    except Exception as e:
        return {"error": str(e)}


# ------------------ Fallbacks / NL parsing ------------------

def generic_profile() -> dict:
    """Fallback when user provides no data."""
    return {
        "name": None,
        "position": "Recruiter",
        "company": "Mid-sized software company",
        "location": "United States (metro area)",
        "url": None,
    }


def nl_to_json_extractor(text: str) -> dict:
    """
    NL → JSON via Gemini 2.0 Flash.
    Returns a dict with keys: name, position, company, location, url (all may be None).
    Falls back to a tiny regex heuristic if the model is unavailable or returns bad JSON.
    """
    if not text or not text.strip():
        return generic_profile()

    prompt = (
        "Extract a contact profile from the text below.\n"
        "Respond ONLY with JSON in exactly this schema (no extra keys, no prose):\n"
        "{\n"
        '  "name": string|null,\n'
        '  "position": string|null,\n'
        '  "company": string|null,\n'
        '  "location": string|null,\n'
        '  "url": string|null\n'
        "}\n"
        "- If unsure about a field, use null (not empty string).\n"
        "- If multiple people are mentioned, pick the primary one.\n"
        "- If a LinkedIn profile URL is present, put it in url.\n\n"
        f"Text:\n{text}"
    )

    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 256,
            },
        )
        data = json.loads(resp.text or "{}")

        out = {
            "name": data.get("name"),
            "position": data.get("position"),
            "company": data.get("company"),
            "location": data.get("location"),
            "url": data.get("url"),
        }

        # Normalize whitespace/casing lightly
        for k in list(out.keys()):
            v = out[k]
            if isinstance(v, str):
                v = v.strip()
                out[k] = v if v else None

        # If the model gave us an empty/None-only dict, use a gentle fallback
        if all(out.get(k) is None for k in ("name", "position", "company", "location", "url")):
            raise ValueError("Empty JSON from model")

        return out

    except Exception:
        # ---- tiny, non-blocking fallback (keeps your old behavior) ----
        pos = comp = loc = None
        m = re.search(r"\bat\s+([A-Z][\w&.\- ]{1,60})", text)
        if m: comp = m.group(1).strip()
        m = re.search(r"\b(in|based in)\s+([A-Z][\w .,-]{1,60})", text, re.I)
        if m: loc = m.group(2).strip()
        return {
            "name": None,
            "position": None,
            "company": comp,
            "location": loc,
            "url": None,
        }


# ------------------ High-level builder ------------------

def build_profile(input_data: dict) -> dict:
    """
    Build a normalized profile JSON.

    Accepts any of:
      - input_data["image_bytes"]: QR image bytes from ADK Web
      - input_data["linkedin_url"]: string
      - input_data["preferences"]: natural language string

    Resolution priority:
      1) QR image → LinkedIn URL → scrape
      2) LinkedIn URL → scrape
      3) Preferences (NL) → JSON
      4) Generic fallback
    """
    # 1) QR image → LinkedIn
    image_bytes: Optional[bytes] = input_data.get("image_bytes")
    if image_bytes:
        qr = qr_to_vcard_or_url(image_bytes=image_bytes, mime_type=input_data.get("mime_type"))
        if qr.get("status") == "success":
            input_data["linkedin_url"] = qr["linkedin_url"]
        else:
            # Hard fail on non-LinkedIn QR per your requirement
            return {"error": qr.get("message", "Invalid QR code")}

    # 2) LinkedIn URL
    if input_data.get("linkedin_url"):
        return linkedin_profile_extractor(
            linkedin_url=input_data["linkedin_url"],
            email=input_data.get("email"),
            password=input_data.get("password"),
            headless=input_data.get("headless", True),
            chromedriver=input_data.get("chromedriver"),
        )

    # 3) Natural language preferences
    if input_data.get("preferences"):
        return nl_to_json_extractor(input_data["preferences"])

    # 4) Fallback
    return generic_profile()


# ============================================================
# ===============   ADK-friendly tool wrappers   =============
# ============================================================

def google_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Perform a Google search to get current information.
    Useful when agents need to verify facts or get recent company info.
    """
    try:
        from googlesearch import search

        results = []
        search_results = search(query, num_results=num_results, sleep_interval=1)

        for i, url in enumerate(search_results):
            if i >= num_results:
                break
            results.append({
                "title": f"Result {i + 1}",
                "url": url,
                "snippet": f"Search result for: {query}"
            })

        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        }

    except ImportError:
        # Fallback if googlesearch package isn't installed
        return {
            "status": "error",
            "message": "Google search not available. Install with: pip install googlesearch-python",
            "fallback_advice": f"For query '{query}', suggest checking the company's official website, LinkedIn, or recent news articles."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Search failed: {str(e)}",
            "query": query
        }


def ping() -> Dict[str, Any]:
    """Simple smoke test tool so you can verify calls work in the dev UI."""
    return {"ok": True, "message": "pong"}


def parse_qr_b64(image_b64: str, mime_type: str = "image/png") -> Dict[str, Any]:
    """
    Accepts a Base64-encoded image (from ADK Web) and feeds it to qr_to_vcard_or_url.
    """
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception as e:
        return {"status": "error", "message": f"Invalid base64: {e}"}
    return qr_to_vcard_or_url(image_bytes=image_bytes, mime_type=mime_type)


def build_profile_from_inputs(
        image_b64: Optional[str] = None,
        linkedin_url: Optional[str] = None,
        preferences: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        headless: bool = True,
        chromedriver: Optional[str] = None,
        mime_type: str = "image/png",
) -> Dict[str, Any]:
    """
    Unified entrypoint: provide any of image_b64, linkedin_url, or preferences.
    """
    payload: Dict[str, Any] = {
        "linkedin_url": linkedin_url,
        "preferences": preferences,
        "email": email,
        "password": password,
        "headless": headless,
        "chromedriver": chromedriver,
        "mime_type": mime_type,
    }
    if image_b64:
        try:
            payload["image_bytes"] = base64.b64decode(image_b64, validate=True)
        except Exception as e:
            return {"error": f"Invalid base64 image: {e}"}
    return build_profile(payload)


def render_prompt_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    return render_prompt_from_json(profile or {})


# ============================================================
# =============   STATE-BASED AGENT COMMUNICATION   ==========
# ============================================================

def broadcast_prompt_to_agents(prompt_dict: dict, state: dict) -> str:
    """
    Store the generated prompt in shared state for other agents to pick up.
    This replaces the send_to functionality for ADK 1.15.1
    """
    if not prompt_dict or "persona_instructions" not in prompt_dict:
        return "❌ Error: No valid prompt to broadcast."

    # Store in shared state that other agents can access
    state["shared_prompt"] = {
        "instructions": prompt_dict["persona_instructions"],
        "profile": prompt_dict.get("profile", {}),
        "timestamp": json.dumps({"time": "now"}),  # Simple timestamp
        "ready": True
    }

    # Also store a message queue for agents to process
    messages = state.setdefault("agent_messages", {})
    messages["persona_agent"] = {
        "type": "new_persona",
        "data": prompt_dict["persona_instructions"],
        "profile": prompt_dict.get("profile", {})
    }
    messages["coach_agent"] = {
        "type": "coaching_context",
        "data": f"New persona created: {prompt_dict.get('profile', {}).get('name', 'Unknown')} at {prompt_dict.get('profile', {}).get('company', 'Unknown Company')}",
        "profile": prompt_dict.get("profile", {})
    }

    profile_name = prompt_dict.get("profile", {}).get("name", "Unknown")
    profile_company = prompt_dict.get("profile", {}).get("company", "Unknown Company")

    return f"✅ Prompt broadcasted to persona_agent and coach_agent!\n\nPersona: {profile_name} at {profile_company}\n\nBoth agents now have access to the new persona instructions via shared state."


# ============================================================
# =====================    root_agent    =====================
# ============================================================

root_agent = Agent(
    name="delegator_smoke_agent",
    model="gemini-2.0-flash",
    description="Turns LinkedIn URLs, QR codes, or natural language into recruiter profiles and sends them to other agents.",
    instruction=(
        "You are the root delegator agent. Your job is to:\n"
        "1. Process user input (LinkedIn URL, QR code, or natural language)\n"
        "2. Build a profile using the appropriate tool\n"
        "3. Generate persona instructions from the profile\n"
        "4. Broadcast the prompt to other agents via shared state\n\n"
        "Examples:\n"
        "User: https://www.linkedin.com/in/jaylynj\n"
        "→ Call build_profile_from_inputs(linkedin_url=...)\n"
        "→ Call render_prompt_from_profile(profile=...)\n"
        "→ Call broadcast_prompt_to_agents(prompt_dict=...)\n\n"
        "User: [uploads QR code image]\n"
        "→ Call parse_qr_b64(image_b64=...) first to extract URL\n"
        "→ Then follow LinkedIn URL flow\n\n"
        "User: I met a software engineer at Google based in San Francisco\n"
        "→ Call build_profile_from_inputs(preferences=...)\n"
        "→ Call render_prompt_from_profile(profile=...)\n"
        "→ Call broadcast_prompt_to_agents(prompt_dict=...)\n\n"
        "If you need current information about a company or role, use google_search.\n"
        "Always complete the full pipeline: input → profile → prompt → broadcast."
    ),
    tools=[
        parse_qr_b64,
        build_profile_from_inputs,
        render_prompt_from_profile,
        broadcast_prompt_to_agents,
        google_search,
        generic_profile,
        ping,
    ],
)