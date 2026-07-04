from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent


# ------------------------------------------
# STEP 1: Inicializar el modelo
# ------------------------------------------

model = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)


# ------------------------------------------
# STEP 2: Definir las herramientas
# ------------------------------------------

@tool
def add(a: float, b: float) -> float:
    """Suma dos números."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiplica dos números."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide dos números."""
    return a / b

tools = [add, multiply, divide]


# ------------------------------------------
# STEP 3: Crear el agente
# ------------------------------------------

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "Eres un asistente matemático. "
        "Usa las herramientas disponibles para sumar, multiplicar y dividir."
    ),
)


# ------------------------------------------
# STEP 4: Ejecutar el agente
# ------------------------------------------

def run_agent(question: str) -> None:
    print(f"\nUsuario: {question}")
    print("-" * 50)

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question}
            ]
        }
    )

    # Muestra los mensajes, llamadas a herramientas y resultados.
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    while True:
        question = input("\nPregunta (o 'salir'): ")

        if question.lower() == "salir":
            break

        run_agent(question)