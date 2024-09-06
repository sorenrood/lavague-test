import os
from lavague.contexts.openai import OpenaiContext
from lavague.core.agents import WebAgent
from lavague.drivers.selenium import SeleniumDriver
from lavague.core.world_model import WorldModel
from lavague.core.action_engine import ActionEngine

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize context and agent
context = OpenaiContext(
    api_key=api_key,
    llm="gpt-4o-mini",
    mm_llm="gpt-4o-mini"
)
selenium_driver = SeleniumDriver()
action_engine = ActionEngine.from_context(context=context, driver=selenium_driver)
world_model = WorldModel.from_context(context)
agent = WebAgent(world_model, action_engine)

# Run query
agent.get("https://huggingface.co/")
result = agent.run("What is this week's top Space of the week?")
print(result)