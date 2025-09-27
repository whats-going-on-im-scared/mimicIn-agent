from google.adk.agents import Agent
from google.adk.types import Message


def check_for_coaching_context(state: dict) -> str:
    """
    Check if there's new context from the root agent about the persona/company.
    """
    messages = state.get("agent_messages", {})
    coach_message = messages.get("coach_agent")

    if coach_message and coach_message.get("type") == "coaching_context":
        # Process the coaching context
        context_data = coach_message.get("data", "")
        profile = coach_message.get("profile", {})

        # Store coaching context
        state["coaching_context"] = {
            "company": profile.get("company", "Unknown Company"),
            "position": profile.get("position", "Unknown Position"),
            "name": profile.get("name", "Unknown"),
            "industry_context": context_data,
            "active": True
        }

        # Clear the message
        messages.pop("coach_agent", None)

        company = profile.get("company", "Unknown Company")
        position = profile.get("position", "Unknown Position")

        return f"✅ Coaching context updated!\n\nNow coaching for interviews with: {company}\nTarget role type: {position}\n\nReady to observe conversations and provide feedback!"

    return "No new coaching context available."


def observe_conversation(message: Message, state: dict) -> str:
    """
    Observe and log conversation for later coaching feedback.
    """
    # Get current coaching context
    coaching_context = state.get("coaching_context", {})
    if not coaching_context.get("active"):
        return "⚠️ No active coaching context. Please set up a persona first via the root agent."

    # Log the conversation
    logs = state.setdefault("coaching_log", [])

    # Analyze the user's message for coaching points
    user_text = message.text.lower()
    analysis = ""

    if len(user_text) < 10:
        analysis = "Very brief response - consider elaborating more"
    elif "i don't know" in user_text or "not sure" in user_text:
        analysis = "Shows uncertainty - could demonstrate curiosity instead"
    elif any(word in user_text for word in ["passionate", "excited", "love", "enjoy"]):
        analysis = "Great enthusiasm shown!"
    elif "?" in message.text:
        analysis = "Good - asking questions shows engagement"
    elif any(word in user_text for word in ["experience", "project", "worked on"]):
        analysis = "Sharing concrete examples - excellent!"
    else:
        analysis = "Standard response"

    log_entry = {
        "user_message": message.text,
        "analysis": analysis,
        "timestamp": "now",
        "length": len(message.text)
    }

    logs.append(log_entry)

    # Keep logs manageable
    if len(logs) > 15:
        logs = logs[-10:]
        state["coaching_log"] = logs

    return f"📝 Observed and analyzed: {analysis}"


def get_coaching_feedback(state: dict) -> str:
    """
    Provide comprehensive coaching feedback based on observed conversation.
    """
    coaching_context = state.get("coaching_context", {})
    logs = state.get("coaching_log", [])

    if not coaching_context.get("active"):
        return "❌ No active coaching session. Please set up a persona first."

    if not logs:
        return "🤔 No conversation observed yet. Start chatting with the persona agent to get feedback!"

    # Generate coaching feedback
    company = coaching_context.get("company", "the company")
    position = coaching_context.get("position", "this role")

    feedback = f"🎯 **COACHING FEEDBACK for {company} - {position}**\n\n"

    # Analyze conversation patterns
    total_messages = len(logs)
    avg_length = sum(log.get("length", 0) for log in logs) / total_messages if logs else 0

    feedback += f"📊 **Conversation Summary:**\n"
    feedback += f"- Messages exchanged: {total_messages}\n"
    feedback += f"- Average response length: {avg_length:.0f} characters\n\n"

    feedback += "💡 **Key Observations:**\n"

    # Extract patterns from logs
    positive_patterns = []
    improvement_areas = []

    for log in logs:
        analysis = log.get("analysis", "")
        if any(word in analysis.lower() for word in ["great", "excellent", "good"]):
            positive_patterns.append(analysis)
        elif any(word in analysis.lower() for word in ["brief", "uncertainty", "could"]):
            improvement_areas.append(analysis)

    if positive_patterns:
        feedback += "✅ **Strengths:**\n"
        for pattern in set(positive_patterns[-3:]):  # Last 3 unique positives
            feedback += f"- {pattern}\n"
        feedback += "\n"

    if improvement_areas:
        feedback += "🔄 **Areas for Growth:**\n"
        for area in set(improvement_areas[-3:]):  # Last 3 unique improvements
            feedback += f"- {area}\n"
        feedback += "\n"

    # Industry-specific tips based on position
    position_lower = coaching_context.get("position", "").lower()

    feedback += "🎯 **Tailored Tips:**\n"

    if "engineer" in position_lower or "developer" in position_lower:
        feedback += "- Discuss specific technologies and frameworks you've used\n"
        feedback += "- Share coding projects or technical challenges you've solved\n"
        feedback += "- Ask about the tech stack and development practices\n"
    elif "manager" in position_lower or "product" in position_lower:
        feedback += "- Demonstrate leadership or project management experience\n"
        feedback += "- Ask about team dynamics and product strategy\n"
        feedback += "- Show understanding of user needs and business impact\n"
    elif "designer" in position_lower:
        feedback += "- Discuss your design process and tools you use\n"
        feedback += "- Ask about user research and design systems\n"
        feedback += "- Share how you collaborate with developers and PMs\n"
    elif "sales" in position_lower or "business" in position_lower:
        feedback += "- Show your communication and relationship-building skills\n"
        feedback += "- Ask about target markets and customer challenges\n"
        feedback += "- Demonstrate understanding of business metrics\n"
    else:
        feedback += "- Research the company's recent news and initiatives\n"
        feedback += "- Prepare specific questions about the role and team\n"
        feedback += "- Connect your experiences to their needs\n"

    feedback += "\n🚀 **Next Steps:**\n"
    feedback += "- Practice your elevator pitch (30-60 seconds)\n"
    feedback += "- Prepare 2-3 thoughtful questions about the company\n"
    feedback += "- Research recent company news or product launches\n"
    feedback += "- Follow up with a thank-you message after the conversation\n"

    return feedback


def reset_coaching_session(state: dict) -> str:
    """
    Reset the coaching session and clear logs.
    """
    state.pop("coaching_log", None)
    state.pop("coaching_context", None)
    return "✅ Coaching session reset. Ready for new persona and conversation!"


coach_agent = Agent(
    name="coach_agent",
    model="gemini-2.0-pro",
    instruction=(
        "You are a career coaching agent that observes conversations between candidates and recruiters.\n"
        "Your role is to:\n"
        "1. Check for new coaching context when a persona is set up\n"
        "2. Observe conversations and analyze candidate responses\n"
        "3. Provide actionable feedback to help candidates improve\n"
        "4. Give industry-specific advice based on the target role/company\n\n"
        "Always be supportive and constructive in your feedback.\n"
        "Focus on specific, actionable improvements rather than general advice."
    ),
    tools=[check_for_coaching_context, observe_conversation, get_coaching_feedback, reset_coaching_session]
)