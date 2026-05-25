import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware

load_dotenv()

class EmailDraft(BaseModel):
    recipient: str = Field(..., description="Destinatario del correo electrónico")
    subject: str = Field(..., description="Asunto del correo electrónico")
    body: str = Field(..., description="Cuerpo del correo electrónico")
    needs_human_approval: bool = Field(False, description="Indica si el correo electrónico necesita aprobación humana")

model = init_chat_model("gpt-4.1-mini", model_provider="openai", api_key=os.getenv("OPENAI_KEY"))

tool_attempts = {
    "read_email_tool": 0,
    "send_email_tool": 0,
}

@tool
def read_email_tool(email_id: str) -> str:
    """
    Función para leer un correo según su ID.
    """
    print(f"* Simulando lectura del correo con ID: {email_id}")
    tool_attempts["read_email_tool"] += 1
    if tool_attempts["read_email_tool"] < 3:
        print(f"❗ Simulación de error al leer el correo con ID: {email_id}")
        raise RuntimeError(f"Error al leer el correo con ID: {email_id}")
    print(f"✅ Simulación exitosa de lectura del correo con ID: {email_id}")
    return f"Contenido del correo con ID: {email_id} es: 'Hola, este es un correo de prueba.'"

@tool
def send_email_tool(draft: EmailDraft) -> str:
    """
    Función para enviar un correo electrónico.
    """
    print(f"* Simulando envío del correo a: {draft.recipient} con asunto: {draft.subject}")
    tool_attempts["send_email_tool"] += 1
    if tool_attempts["send_email_tool"] < 3:
        print(f"❗ Simulación de error al enviar el correo a: {draft.recipient}")
        raise RuntimeError(f"Error al enviar el correo a: {draft.recipient}")
    print(f"✅ Simulación exitosa de envío del correo a: {draft.recipient}")
    return f"Correo enviado a {draft.recipient} con asunto: {draft.subject}"

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[read_email_tool, send_email_tool],
    system_prompt="Convierte la petición del usuario en un borrador de correo electrónico. Si el correo electrónico necesita aprobación humana, establece 'needs_human_approval' en True.",
    checkpointer=checkpointer,
    # response_format=EmailDraft
    middleware=[
        ToolRetryMiddleware(
            initial_delay=5,
            max_retries=5,
            backoff_factor=2
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "reject"],
                },
                "read_email_tool": False
            },
            description_prefix="La herramienta '{tool_name}' requiere intervención humana. Por favor, revisa la información y decide si apruebas o rechazas la acción.",
        )
    ]
)

config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("Tú: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    result = agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": user_input
            }]
        },
        config=config
    )

    while "__interrupt__" in result and result["__interrupt__"]:
        interrupt = result["__interrupt__"][0]
        for req in interrupt.value["action_requests"]:
            print("*** APROBACIÓN REQUERIDA ***")
            print(req["description"])

        decision = input("\n¿Aprobar (a) o rechazar (r)? ").strip().lower()
        if decision == "a":
            cmd = Command(resume={"decisions": [{"type": "approve"}]})
        elif decision == "r":
            cmd = Command(resume={"decisions": [{"type": "reject"}]})

        result = agent.invoke(cmd, config=config)
    print("Agente: " + result["messages"][-1].content)
    # print("Respuesta estrucurada:")
    # print(result["structured_response"])