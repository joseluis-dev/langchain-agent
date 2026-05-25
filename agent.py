import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = init_chat_model("gpt-4.1-mini", model_provider="openai", api_key=os.getenv("OPENAI_KEY"))

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    data = {
        "ambato": "27°C, sunny",
        "quito": "22°C, cloudy",
        "guayaquil": "30°C, sunny",
    }
    return data.get(city.lower(), "City not found")

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="Eres un asistente que habla español, y tus respuestas son muy concisas y breves.",
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "1"}}

agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Mi nombre es Jose"
        }]
    },
    config=config
)

result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "¿Cuál es mi nombre?"
        }]
    },
    config=config
)

print(result["messages"][-1].content)