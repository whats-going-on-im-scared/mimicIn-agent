from google.adk.agents import Agent

def hello_world() -> str:
    """Outputs 'hello world'
    Returns:
        str: hello world
    """
    return "Hello World"

root_agent = Agent(
    name="demo_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent that says hello world."
    ),
    instruction=(
        "You are an agent who only responds to questions by using the output of your hello_world tool."
    ),
    tools=[hello_world]
)