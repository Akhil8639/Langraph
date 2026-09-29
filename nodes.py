import time

from langchain_core.messages import ToolMessage

from react import llm_with_tools, tools


# ============================================================
# SYSTEM MESSAGE
# ============================================================

SYSTEM_MESSAGE = """
You are a helpful ReAct-style AI agent.

You can use tools when necessary.

If the user asks for current information,
use the appropriate search tool.

If mathematical calculation is required,
use the multiply tool.

Think about what tool is required,
observe the tool result,
and then provide the final answer.
"""


# ============================================================
# TOOL LOOKUP
# ============================================================

tools_by_name = {
    tool.name: tool
    for tool in tools
}


# ============================================================
# AGENT REASONING NODE
# ============================================================

def run_agent_reasoning(state):

    max_retries = 3

    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        *state["messages"],
    ]

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            response = llm_with_tools.invoke(
                messages
            )

            return {
                "messages": [response]
            }

        except Exception as e:

            error_message = str(e)

            # Handle temporary Gemini 503 error
            if (
                "503" in error_message
                or "UNAVAILABLE" in error_message
            ):

                print(
                    f"\nGemini is busy. "
                    f"Retry {attempt}/{max_retries}"
                )

                if attempt < max_retries:

                    wait_time = attempt * 5

                    print(
                        f"Waiting {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:

                    print(
                        "Gemini is still unavailable."
                    )

                    raise

            else:

                raise


# ============================================================
# TOOL EXECUTION NODE
# ============================================================

def tool_node(state):

    last_message = state["messages"][-1]

    tool_messages = []

    for tool_call in last_message.tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]

        tool_call_id = tool_call["id"]

        print(
            f"\nCalling tool: {tool_name}"
        )

        print(
            f"Arguments: {tool_args}"
        )

        tool_to_use = tools_by_name[
            tool_name
        ]

        # Execute tool
        observation = tool_to_use.invoke(
            tool_args
        )

        print(
            f"Observation: {observation}"
        )

        # Return tool result to LLM
        tool_message = ToolMessage(
            content=str(observation),
            tool_call_id=tool_call_id,
        )

        tool_messages.append(
            tool_message
        )

    return {
        "messages": tool_messages
    }