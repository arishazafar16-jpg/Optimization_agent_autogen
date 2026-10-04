import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

async def main():
    model_client = OllamaChatCompletionClient(model="llama3.2")

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
    )

    response = await assistant.on_messages(
        [TextMessage(content="What's 2+2? Explain your reasoning.", source="user")],
        cancellation_token=None,
    )

    print(response.chat_message.content)

asyncio.run(main())


