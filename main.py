from asyncio import run
import subprocess
from typing import Literal
from agents import (
    Agent,
    ModelBehaviorError,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    enable_verbose_stdout_logging,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import pydantic_core
from syntaxmod import Stopwatch, Timer


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
    tech_stack_specific: list[Service] = Field(...)
    "specific service. eg. Next.js, react, Python, OpenAI, etc."


enable_verbose_stdout_logging()

with open("./web.log", "w"):
    pass

stopwatch = Stopwatch()

@function_tool
def elapsed_time():
    return stopwatch.get_elasped_time()

@function_tool
async def bash_command(full_command: str) -> tuple[str, str]:
    global tool_uses
    if full_command.startswith("search "):
        inp = f"browser-use open 'https://www.google.com/?q={full_command.replace('search ', '')} && browser-use state"

    else:
        inp = full_command
    cmd: subprocess.CompletedProcess[str] = subprocess.run(
        args=inp, capture_output=True, text=True, shell=True
    )
    print(full_command)
    with open("./web.log", "a") as f:
        f.write(f"\nTOOL CALLED: {repr(full_command)}")

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
When sayign the tech stack/ Service, be specific
Don't just say, frontend, backend, core engine
Say Next.js, python with Flask, and OpenAI for example, and use whichever specific stack you want
this is important
Be very specific
Don't jsut say frontend service
say which one
so like we have Next.js, react, svelte, etc.
same goes for evrything else
This is a great time to search it up
USE YOUR TOOLS
USE YOUR TOOLS
USE YOUR TOOLS
SEARCH EVRYTHIG UP
YOU ARE A MASSIVE FALIURE IF YOU DONT SEARCH VERYHTING USING BASH_COMMAND and browser-use
USE YOUR TOOLS
STOP IGNORING THEM
I can see if you used your tools or not
be very verbose about when and whch tools you used
search up everything
Your other tool is a chatbot
GIVE IT FULL CONTEXT BEFORE JUST ASKING IT
Use it to validate all of your ideas
Don't make a single decision until it has approved
It will try to prove you wroong, and ONLY if it cant it willl agree
It will alwauys point out evrything wrong with your idea
It will validate everythig with web search and broswer-use 
try to stay under 20 minutes, its okay if you go over
to check how much time has elapsed, you can use elapsed_time, the tool

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

cagent_instruction = INSTRUCTIONS
cagent_instruction += "\nAll of these apply, except you are the chatbot in question"

Cagent = Agent(
    "gemma4Chat1",
    model=model,
    tools=[bash_command, elapsed_time],
    instructions=INSTRUCTIONS,
    output_type=Business,
)

agent = Agent(
    "gemma4Chat1",
    model=model,
    tools=[bash_command, Cagent.as_tool("Chat Agent", "Use for chats"), elapsed_time],
    instructions=INSTRUCTIONS,
    output_type=Business,
)

async def chat(input: str, context: str):
    put = await Runner().run(Cagent, input=f"CONTEXT: {context} INPUT: {input}")
    return put.final_output

async def main():
    while True:
        BANNED_SERVICE_PHRASES = [
            "Backend",
            "Backend ServiceDatabase",
            "Database ServiceFrontend",
            "Frontend Service",
        ]
        try:
            output = await Runner().run(
                agent,
                "idea for saas company, you make it up. search up as much stuff as possible.",
                max_turns=None,
            )
            print(output.final_output)
            business: Business = output.final_output
            if [i.name in BANNED_SERVICE_PHRASES for i in Business.tech_stack_specific]:
                raise ValueError

            print(f"""The name of the app is {business.name}
        here is a description: {business.description}
        here is the tech stack: {[i.name for i in business.tech_stack_specific]}""")
            return business

        except ModelBehaviorError, ValueError, AttributeError:
            continue



idea = run(main())

terminator = Agent("Big Boi", tools=[bash_command])

class Idea(BaseModel):
    good_idea: Literal["Yes", "No"]
    what_could_be_better: list[str]