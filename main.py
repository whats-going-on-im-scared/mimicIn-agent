import os
import json
import asyncio
import base64
import warnings

from pathlib import Path
from dotenv import load_dotenv

from google.genai.types import (
    Part,
    Content,
    Blob,
)

from google.adk.runners import InMemoryRunner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.genai import types

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from converse.agent import root_agent

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

#
# ADK Streaming
#

# Load Gemini API Key
load_dotenv()

APP_NAME = "Testing server for websockets"


async def start_agent_session(user_id, is_audio=False):
    """Starts an agent session"""

    # Create a Runner
    runner = InMemoryRunner(
        app_name=APP_NAME,
        agent=root_agent,
    )

    # Create a Session
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        state={
            "prompt":"""
                You are Linda Arlet, the Vice President and Senior Recruiter for Investment Services at BNY,\n    located in the Greater Pittsburgh Region. You are interviewing a candidate at a job fair who has\n    expressed interest in joining the firm. Please ask questions that Linda would be likely to ask given\n    her summary: Her career is deeply rooted in recruitment and human resources, with extensive experience\n    within major financial institutions. She holds both a Master's and a Bachelor's degree in Human Resource\n    Management. ### **Professional Summary** * **Current Role**: Ms. Arlet is a Vice President and Senior\n    Recruiter at BNY, a position she has held since January 2022. Her activity on LinkedIn shows she is\n    actively recruiting for senior-level roles within the company, such as \"Vice President, Business\n    Continuity & Recovery\" and \"VP, Core Clearing Product Owner\". * **Company**: BNY (The Bank of\n    New York Mellon) is a global investments company focused on managing and servicing assets for institutions,\n    corporations, and individual investors. * **Experience**: She has a long history in the financial services\n    industry, having spent nearly six years as a Specialist Recruiter for Wealth Management at Citizens\n    Financial Group, Inc., and over ten years at PNC Financial Services Group in both Recruiter and Employee\n    Relations Consultant roles. * **Education**: Linda Arlet earned a Master of Arts in Human Resource\n    Management from St. Francis University and a Bachelor of Science in HR Management from Robert Morris\n    University. ### **Detailed Information from Profile** **Career Experience:*** **Vice President,\n    Senior Recruiter, Investment Services** at BNY (Jan 2022 - Present).  * **Specialist Recruiter,\n    Wealth Management** at Citizens Financial Group, Inc. (Apr 2016 - Feb 2022). * **Recruiter** at PNC\n    Financial Services Group (Nov 2007 - Apr 2016). * **Employee Relations Consultant** at PNC Financial\n    Services Group (Apr 2006 - Nov 2007).  **Licenses & Certifications:** * **Nano Tips to Foster a\n    Growth Mindset and Mental Agility with Shadé Zahrai** - Issued by LinkedIn in April 2025.\n    **Recruitment Focus**: Her posts indicate she recruits for early career positions in finance,\n    technology, and risk management, specifically for locations in Lake Mary, FL, and Pittsburgh, PA.\n    **Industry Engagement**: She follows other major financial institutions like UBS and KeyBank, and\n    is a member of professional groups such as \"Finance & Banking, Fintech, Regtech Professionals\n    Worldwide\" and \"Anti-Money Laundering Specialists\". **Professional Interests**: She follows\n    Top Voices on LinkedIn including her company's CEO, Robin Vince, and financial personality Suze\n    Orman, suggesting an interest in leadership and financial trends. **Peer Relationships**: She has\n    given a recommendation to a former colleague from PNC, describing her as a \"hard worker\" with\n    a \"positive attitude\".
            """
        }  # Replace with actual user ID
    )

    # Set response modality
    modality = "AUDIO" if is_audio else "TEXT"
    run_config = RunConfig(
        response_modalities=[modality],
        session_resumption=types.SessionResumptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    # Create a LiveRequestQueue for this session
    live_request_queue = LiveRequestQueue()

    # Start agent session
    live_events = runner.run_live(
        session=session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )
    return live_events, live_request_queue


async def agent_to_client_messaging(websocket, live_events):
    """Agent to client communication"""
    async for event in live_events:

        # If the turn complete or interrupted, send it
        if event.turn_complete or event.interrupted:
            message = {
                "turn_complete": event.turn_complete,
                "interrupted": event.interrupted,
            }
            await websocket.send_text(json.dumps(message))
            print(f"[AGENT TO CLIENT]: {message}")
            continue

        # Read the Content and its first Part
        part: Part = (
            event.content and event.content.parts and event.content.parts[0]
        )
        if not part:
            continue

        # If it's audio, send Base64 encoded audio data
        is_audio = part.inline_data and part.inline_data.mime_type.startswith("audio/pcm")
        if is_audio:
            audio_data = part.inline_data and part.inline_data.data
            if audio_data:
                message = {
                    "mime_type": "audio/pcm",
                    "data": base64.b64encode(audio_data).decode("ascii")
                }
                await websocket.send_text(json.dumps(message))
                print(f"[AGENT TO CLIENT]: audio/pcm: {len(audio_data)} bytes.")
                continue

        # If it's text and a partial text, send it
        if part.text and event.partial:
            message = {
                "mime_type": "text/plain",
                "data": part.text
            }
            await websocket.send_text(json.dumps(message))
            print(f"[AGENT TO CLIENT]: text/plain: {message}")


async def client_to_agent_messaging(websocket, live_request_queue):
    """Client to agent communication"""
    while True:
        # Decode JSON message
        message_json = await websocket.receive_text()
        message = json.loads(message_json)
        mime_type = message["mime_type"]
        data = message["data"]

        # Send the message to the agent
        if mime_type == "text/plain":
            # Send a text message
            content = Content(role="user", parts=[Part.from_text(text=data)])
            live_request_queue.send_content(content=content)
            print(f"[CLIENT TO AGENT]: {data}")
        elif mime_type == "audio/pcm":
            # Send an audio data
            decoded_data = base64.b64decode(data)
            live_request_queue.send_realtime(Blob(data=decoded_data, mime_type=mime_type))
        else:
            raise ValueError(f"Mime type not supported: {mime_type}")


#
# FastAPI web app
#

app = FastAPI()

STATIC_DIR = Path("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """Serves the index.html"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, is_audio: str):
    """Client websocket endpoint"""

    # Wait for client connection
    await websocket.accept()
    print(f"Client #{user_id} connected, audio mode: {is_audio}")

    # Start agent session
    user_id_str = str(user_id)
    live_events, live_request_queue = await start_agent_session(user_id_str, is_audio == "true")

    # Start tasks
    agent_to_client_task = asyncio.create_task(
        agent_to_client_messaging(websocket, live_events)
    )
    client_to_agent_task = asyncio.create_task(
        client_to_agent_messaging(websocket, live_request_queue)
    )

    # Wait until the websocket is disconnected or an error occurs
    tasks = [agent_to_client_task, client_to_agent_task]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Close LiveRequestQueue
    live_request_queue.close()

    # Disconnected
    print(f"Client #{user_id} disconnected")