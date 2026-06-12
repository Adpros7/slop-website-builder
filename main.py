from asyncio import run
import subprocess
from typing import Literal

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import pydantic_core


client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
model = OpenAIChatCompletionsModel("gemma4", client)

with open("SKILL.md", "r") as f:
    skill = f.read()


class Service(BaseModel):
    name: str
    usage: str = Field(
        ...,
        description="where to use. for example, python backeend. openai at the backend, etc.",
    )


class Business(BaseModel):
    name: str
    description: str
    long_description: str
    architecture: str
    other_info: str
    tech_stack: list[Service] = Field(..., description="specific service. eg. Next.js")
    "specific service. eg. Next.js, react, Python, OpenAI, etc."


@function_tool
async def bash_command(full_command: str) -> tuple[str, str]:
    global tool_uses
    cmd: subprocess.CompletedProcess[str] = subprocess.run(
        args=full_command, capture_output=True, text=True, shell=True
    )
    print(full_command)
    return cmd.stdout, cmd.stderr


INSTRUCTIONS = f"""You are an expert who comes up with buisness ideas that will make a lot of money
You use all of your tools
You search the Web for existing solutions
You find an original Problem to solve
You detail how to solve it
You make sure there are no other companies called what you are going to name it
You find a lot of ideas at first, but then narrow it down to just a few, then one
You make up everything
Don't talk to the user
Just do what you are told
Default to a digital application

here is how to search the web:
If you don't use tools, you are wrong
Mandatory tool uses of at least 5
 ALways keep trying at least 5 times with different combinations and things, and use the browser-use skill.
Look at your tool options before using any. Be very verbose about which tools you used, why, and if they failed. 
 Use bash to run browser-use commands. 
 browser-use state is a really important command. 
 When opening urls using browser-use always surround the url in single quoutes.
 After running browser-use open * run browser-use state
 MOST IMPORTANT PART HERE IS HOW TO USE BROWSER USE SKILL: browser use skill: {skill} THIS IS THE MLOST IMPORTANT THING THIS IS DOCUMENTATION FOR browser-use
"""

agent = Agent("gemma4Chat1", model=model, output_type=Business, tools=[bash_command])


async def main():
    output = await Runner().run(agent, "idea for saas company, you make it up")
    business: Business = output.final_output
    print(f"""The name of the app is {business.name}
here is a description: {business.description}
here is the tech stack: {[i.name for i in business.tech_stack]}""")


run(main())
