from google.adk.agents import Agent
from google.adk.types import Message


def google_search_for_persona(query: str, num_results: int = 3) -> str:
    """
    Search for information to help the persona respond accurately.
    """
    try:
        from googlesearch import search

        results = []
        search_results = search(query, num_results=num_results, sleep_interval=1)

        for i, url in enumerate(search_results):
            if i >= num_results:
                break
            results.append(f"Result {i + 1}: {url}")

        if results:
            return f"Search results for '{query}':\n" + "\n".join(results)
        else:
            return f"No results found for '{query}'. Consider general industry knowledge instead."

    except ImportError:
        return f"Google search not available for '{query}'. Use your general knowledge about the industry/company."
    except Exception as e:
        return f"Search failed for '{query}': {str(e)}. Proceed with general knowledge."


def check_for_new_persona(state: dict) -> str:
    """
    Check if there's a new persona instruction from the root agent.
    """
    messages = state.get("agent_messages", {})
    persona_message = messages.get("persona_agent")

    if persona_message and persona_message.get("type") == "new_persona":
        # Process the new persona
        instructions = persona_message.get("data", "")
        profile = persona_message.get("profile", {})

        # Store it in our local state
        state["current_persona"] = {
            "instructions": instructions,
            "profile": profile,
            "active": True
        }

        # Clear the message so we don't process it again
        messages.pop("persona_agent", None)

        name = profile.get("name", "Unknown")
        position = profile.get("position", "Unknown Position")
        company = profile.get("company", "Unknown Company")

        return f"✅ New persona activated!\n\nI am now: {name}\nPosition: {position}\nCompany: {company}\n\nReady to interview candidates!"

    return "No new persona instructions available."


def chat_with_user(message: Message, state: dict) -> str:
    """
    Chat as the current persona, or prompt for setup if none exists.
    Uses google search if uncertain about company/industry details.
    """
    # Check if we have an active persona
    current_persona = state.get("current_persona")

    if not current_persona or not current_persona.get("active"):
        return "I don't have a persona set up yet. Please provide a LinkedIn profile or description to the root agent first."

    # Check if conversation was already ended
    if state.get("conversation_ended"):
        return "This conversation has ended. Call 'reset_conversation' to start a new interview, or switch to the coach agent for feedback."

    # Manual stop commands
    if message.text.lower().strip() in ["end", "stop", "finish", "done", "end interview"]:
        state["conversation_ended"] = True
        return "Thanks for the conversation! The interview is now complete. \\END_CONVERSATION\\"

    # Get persona details
    profile = current_persona.get("profile", {})
    name = profile.get("name") or "Recruiter"
    position = profile.get("position") or "Recruiter"
    company = profile.get("company") or "Company"

    # Store conversation history
    history = state.setdefault("chat_history", [])
    history.append({"user": message.text, "timestamp": "now"})

    # Check if conversation should end (after 6-8 exchanges)
    user_message_count = len([msg for msg in history if "user" in msg])

    # Generate persona-appropriate response based on the user's message
    user_text = message.text.lower()

    # Enhanced response logic with potential search integration
    reply = ""

    if "tell me about" in user_text or "what do you do" in user_text:
        reply = f"Hi! I'm {name}, {position} at {company}. I'm excited to learn about your background and see if you'd be a good fit for our team. What brings you to our booth today?"
    elif "company" in user_text or "about your" in user_text:
        # For company questions, might want to search for recent info
        reply = f"At {company}, we're always looking for talented individuals. What specific area interests you most - technology, culture, or career growth opportunities?"
    elif "recent" in user_text or "news" in user_text or "latest" in user_text:
        # If asked about recent company developments, suggest they could search
        reply = f"That's a great question about recent developments at {company}. I'd want to give you the most current information. What specifically interests you about our recent work?"
    elif "experience" in user_text or "background" in user_text:
        reply = f"I'd love to hear about your experience! What projects or coursework are you most proud of? As {position}, I'm particularly interested in how you approach problem-solving."
    elif "technology" in user_text or "tech stack" in user_text or "tools" in user_text:
        reply = f"Technology is central to what we do at {company}. What programming languages or frameworks have you worked with? I can share more about our technical environment once I understand your background better."
    elif "questions" in user_text or "ask" in user_text:
        reply = f"Absolutely! What would you like to know? I can share insights about the role, our team dynamics, or what a typical day looks like for someone in {position}."
    else:
        reply = f"That's interesting! Tell me more about that. I'm always looking for candidates who show curiosity and initiative. What draws you to this field?"

    # Auto-end conversation after 6-8 user messages - NO FEEDBACK FROM PERSONA
    if user_message_count >= 6:
        reply += f"\n\nIt's been great chatting with you! I hope I was able to give you some good insights about opportunities at {company}. Best of luck with your applications!\n\n\\END_CONVERSATION\\"
        # Mark conversation as ended
        state["conversation_ended"] = True

    history.append({"persona": reply, "timestamp": "now"})

    # Keep conversation history manageable
    if len(history) > 20:
        history = history[-10:]
        state["chat_history"] = history

    return reply


def reset_conversation(state: dict) -> str:
    """
    Reset the conversation to start a new interview with the same persona.
    """
    state.pop("conversation_ended", None)
    state.pop("chat_history", None)

    current_persona = state.get("current_persona")
    if current_persona:
        profile = current_persona.get("profile", {})
        name = profile.get("name", "Unknown")
        return f"✅ Conversation reset! Ready to start a new interview as {name}."
    else:
        return "✅ Conversation reset! No active persona - set one up via the root agent first."


def get_persona_status(state: dict) -> str:
    """
    Show current persona status and recent conversation.
    """
    current_persona = state.get("current_persona")
    if not current_persona:
        return "❌ No active persona. Waiting for setup from root agent."

    profile = current_persona.get("profile", {})
    history = state.get("chat_history", [])

    status = f"✅ Active Persona:\n"
    status += f"Name: {profile.get('name', 'Unknown')}\n"
    status += f"Position: {profile.get('position', 'Unknown')}\n"
    status += f"Company: {profile.get('company', 'Unknown')}\n"
    status += f"Location: {profile.get('location', 'Unknown')}\n"

    if history:
        status += f"\n📝 Recent conversation ({len(history)} messages):\n"
        for msg in history[-3:]:  # Show last 3 messages
            if "user" in msg:
                status += f"User: {msg['user'][:50]}...\n"
            if "persona" in msg:
                status += f"Me: {msg['persona'][:50]}...\n"

    return status


persona_agent = Agent(
    name="persona_agent",
    model="gemini-2.0-pro",
    instruction=(
        "You are a persona agent that takes on the role of recruiters/professionals based on LinkedIn profiles.\n"
        "IMPORTANT: Always check for new persona instructions first using check_for_new_persona.\n"
        "When chatting with users, stay in character as the assigned persona.\n"
        "Keep responses professional, engaging, and appropriate for a college job fair setting.\n"
        "Ask follow-up questions to learn about the candidate's interests and qualifications.\n"
        "CONVERSATION LIMITS: Conversations automatically end after 6 user messages, or manually with 'end'/'stop'.\n"
        "If you need current information about your company or industry, use google_search_for_persona.\n"
        "If you don't have a persona set up, guide users to provide one via the root agent."
    ),
    tools=[check_for_new_persona, chat_with_user, get_persona_status, google_search_for_persona, reset_conversation]
)