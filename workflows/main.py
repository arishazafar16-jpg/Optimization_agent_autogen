import asyncio
import os
from autogen_core import CancellationToken
from dotenv import load_dotenv
from scipy.optimize import linprog
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

async def solve_linear_program(
    c: list[float],
    A_ub: list[list[float]],
    b_ub: list[float],
) -> str:
    """Maximize c.x subject to A_ub.x <= b_ub and x >= 0.
    c: objective coefficients, A_ub: constraint matrix, b_ub: constraint limits."""
    result = linprog(
        c=[-v for v in c],
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=(0, None),
    )
    if not result.success:
        return f"Solver failed: {result.message}"
    return f"Optimal x = {list(result.x)}, maximum value = {-result.fun}"

async def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '100 * (2500 + 1.5)'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Only numbers and + - * / ( ) are allowed."
    return str(eval(expression))

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gemini-3.8-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )

    agent = AssistantAgent(
        name="optimizer",
        model_client=model_client,
        tools=[solve_linear_program, calculate],
        system_message=(
            "You are an operations research assistant. Use the calculate tool for "
            "plain arithmetic. Use solve_linear_program only when the problem asks "
            "to maximize or minimize something under constraints. "
            "Always state the final answer clearly with units."
        ),
        reflect_on_tool_use=False,
    )

    print("Optimization Assistant ready. Type 'exit' to quit.")
    while True:
        task = input("\nYou: ")
        if task.strip().lower() == "exit":
            break
        await Console(agent.run_stream(task=task))
        await agent.on_reset(CancellationToken())

    await model_client.close()

asyncio.run(main())