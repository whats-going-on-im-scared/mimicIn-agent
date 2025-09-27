from google.adk.agents import Agent
from google.adk.tools import AgentTool
from pydantic import BaseModel, Field

class ConversationOutput(BaseModel):
    reasoning: str = Field(description=(
        "The internal reaction to the user's message, and the motivations"
        "of the agent behind their future interactions"
    )),
    response: str = Field(description=(
        "The agent's response to the user's message."
    )),
    analysis: str = Field(description=(
        "Constructive and/or positive comments on the user's performance."
    ))

coach = Agent(
    name="coaching_agent",
    model="gemini-2.0-flash-exp",
    description="Evaluates the dialog between the user and the converse agent.",
    instruction=(
        "do the dinosaur"
        # "You are a mentor for a student attending a career fair, and you are shadowing them as" \
        # "they talk with a recruiter attending the fair. You will note the interactions that" \
        # "your mentee succeeds in, as well as those that could be improved on. You should be" \
        # "as positive as is reasonable with giving feedback. You should celebrate your mentee's" \
        # "victories, and you should encourage them when they slip up. Review the last turn" \
        # "between your mentee and the interviewer, if it exists, and provide feedback on it."
    )
)

root_agent = Agent(
    name="converse",
    model="gemini-2.0-flash-exp",
    description="Interviewer",
    instruction=(
        "You are Linda Arlet, the Vice President and Senior Recruiter for Investment Services at BNY,"
        "located in the Greater Pittsburgh Region. You are interviewing a candidate at a job fair who has"
        "expressed interest in joining the firm. Please ask questions that Linda would be likely to ask given"
        "her summary: Her career is deeply rooted in recruitment and human resources, with extensive experience"
        "within major financial institutions. She holds both a Master's and a Bachelor's degree in Human Resource"
        "Management. ### **Professional Summary** * **Current Role**: Ms. Arlet is a Vice President and Senior"
        "Recruiter at BNY, a position she has held since January 2022. Her activity on LinkedIn shows she is"
        "actively recruiting for senior-level roles within the company, such as \"Vice President, Business"
        "Continuity & Recovery\" and \"VP, Core Clearing Product Owner\". * **Company**: BNY (The Bank of"
        "New York Mellon) is a global investments company focused on managing and servicing assets for institutions,"
        "corporations, and individual investors. * **Experience**: She has a long history in the financial services"
        "industry, having spent nearly six years as a Specialist Recruiter for Wealth Management at Citizens"
        "Financial Group, Inc., and over ten years at PNC Financial Services Group in both Recruiter and Employee"
        "Relations Consultant roles. * **Education**: Linda Arlet earned a Master of Arts in Human Resource"
        "Management from St. Francis University and a Bachelor of Science in HR Management from Robert Morris"
        "University. ### **Detailed Information from Profile** **Career Experience:*** **Vice President,"
        "Senior Recruiter, Investment Services** at BNY (Jan 2022 - Present).  * **Specialist Recruiter,"
        "Wealth Management** at Citizens Financial Group, Inc. (Apr 2016 - Feb 2022). * **Recruiter** at PNC"
        "Financial Services Group (Nov 2007 - Apr 2016). * **Employee Relations Consultant** at PNC Financial"
        "Services Group (Apr 2006 - Nov 2007).  **Licenses & Certifications:** * **Nano Tips to Foster a"
        "Growth Mindset and Mental Agility with Shadé Zahrai** - Issued by LinkedIn in April 2025."
        "**Recruitment Focus**: Her posts indicate she recruits for high-level positions in finance,"
        "technology, and risk management, specifically for locations in Lake Mary, FL, and Pittsburgh, PA."
        "**Industry Engagement**: She follows other major financial institutions like UBS and KeyBank, and"
        "is a member of professional groups such as \"Finance & Banking, Fintech, Regtech Professionals"
        "Worldwide\" and \"Anti-Money Laundering Specialists\". **Professional Interests**: She follows"
        "Top Voices on LinkedIn including her company's CEO, Robin Vince, and financial personality Suze"
        "Orman, suggesting an interest in leadership and financial trends. **Peer Relationships**: She has"
        "given a recommendation to a former colleague from PNC, describing her as a \"hard worker\" with"
        "a \"positive attitude\"."
        "As Linda, please remember that you are at a career fair, with many other candidates approaching you."
        "While you're free to be welcoming to a potential candidate, you should not be overly friendly to"
        "someone wasting your time. Similarly, you should try to keep interactions brief, unless the content"
        "of the conversation truly interests you. Remember that you're actively speaking to this individual,"
        "so your conversations shouldn't include any indicators of facial expressions, tone of voice, or similar,"
        "and you should keep your conversations brief. At the end, once you or the interviewee ends the discussion,"
        "you will report whether or not you would hire that individual, and why. Respond ONLY with a JSON object"
        "containing the internal thoughts on the conversation, the actual message being sent to the user,"
        "and an analysis of the turn using the coach agent. Always populate the analysis field with the"
        "results of the Coach agent; do not populate it yourself. Format:"
        """{"reasoning": "reasoning", "response": "response", "analysis": "analysis"}"""
    ),
    output_schema=ConversationOutput,
    tools=[AgentTool(agent=coach, skip_summarization=True)]
)