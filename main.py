from typing import Annotated, TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.graph.message import add_messages

from nodes import (
    run_agent_reasoning,
    tool_node,
)


# ============================================================
# 1. DEFINE STATE
# ============================================================

class AgentState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]


# ============================================================
# 2. ROUTING FUNCTION
# ============================================================

def should_continue(state: AgentState):

    last_message = state["messages"][-1]

    # If model requested tools
    if getattr(
        last_message,
        "tool_calls",
        None,
    ):

        if len(last_message.tool_calls) > 0:

            return "tools"

    # Otherwise finish
    return END


# ============================================================
# 3. CREATE GRAPH
# ============================================================

graph = StateGraph(
    AgentState
)


# ============================================================
# 4. ADD NODES
# ============================================================

graph.add_node(
    "agent_reason",
    run_agent_reasoning,
)

graph.add_node(
    "tools",
    tool_node,
)


# ============================================================
# 5. ADD EDGES
# ============================================================

graph.add_edge(
    START,
    "agent_reason",
)


graph.add_conditional_edges(
    "agent_reason",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)


# After tool executes,
# return to LLM reasoning
graph.add_edge(
    "tools",
    "agent_reason",
)


# ============================================================
# 6. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# 7. RUN
# ============================================================

if __name__ == "__main__":

    print(
        "Hello ReAct LangGraph "
        "with Function Calling"
    )

    query = (
        "What is the temperature in Tokyo? "
        "List it and then triple it."
    )

    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content=query
                )
            ]
        }
    )

    final_message = result[
        "messages"
    ][-1]

    print(
        "\n=============================="
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "==============================\n"
    )

    content = final_message.content

    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                print(part["text"])
    else:
        print(content)