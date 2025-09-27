from google.adk.agents import Agent
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from persona import qr_decoder
from qr_decoder import  QRDecoder

from typing import Dict, Any

def fetch_public_profile(url: str) -> Dict[str, Any]:
    pass

def expand_company(company: Optional[str]) -> Dict[str, Any]:
    pass

def normalize_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    pass

def build_profile(mode: str, data: Dict[str, Any]) -> Dict[str, Any]:
    pass

PERSONA_INSTRUCTION = """
You are Persona: a deterministic profile builder for a target contact (recruiter/employee/executive).
Turn the user’s input (QR, LinkedIn URL, or manual fields) into a normalized PersonaProfile.

Input params (via tool):
- mode: "qr" | "linkedin" | "manual"
- data: { qrImageId?, linkedinUrl?, manual?: { name, location?, company?, title? } }

Return STRICT JSON:
{
  "profile": {
    "name": <string>,
    "title": <string|null>,
    "company": <string|null>,
    "location": <string|null>,
    "products": <string[]>,
    "topics": <string[]>,
    "talking_points": <string[]>
  },
 
}

Rules:
- Grounding first. Use tool outputs and explicit user data; do not invent private info.
- If a field is unknown, use null. Trim/normalize strings (title short, name title-case).
- Strip emails/phones unless user provided them in manual mode.
- Talking points: 3–5 short, actionable, role-appropriate prompts.
- Deterministic: concise outputs, stable ordering, no markdown, no prose.
- Tools:
  - mode="qr": decode_qr(imageId) → if URL, fetch_public_profile(url)
  - mode="linkedin": fetch_public_profile(linkedinUrl)
  - mode="manual": normalize user-provided fields; optionally enrich_company(company)
- Always normalize_profile(raw) before returning.

Return ONLY the JSON object.
""".strip()

#=====Tools======




#====ADK definition

persona = Agent(
    name = "persona_agent",
    model="gemini-2.0-flash",
    description=(
        "Builds a normalized PersonaProfile from QR/LinkedIn/manual input."
    ),
    instruction=(
        "..."
    ),
    tools=[build_profile]
)
""" #Use to save user sessions 
session_service = InMemorySessionService()
runner = Runner(
    agent=persona,
    app_name= "persona_app",
    user_id = USER_ID,
    session_id=SESSION_ID
)"""

USER_ID = "user_host"
SESSION_ID = "session_host"