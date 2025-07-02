from langgraph.graph import StateGraph, START, END

from src.state_models import MainState
from src.nodes import process_files_node, aggregate_content, generate_answer


main_workflow = StateGraph(MainState)
main_workflow.add_node("process_files", process_files_node)
main_workflow.add_node("aggregate_content", aggregate_content)
main_workflow.add_node("generate_answer", generate_answer)

main_workflow.add_edge(START, "process_files")
main_workflow.add_edge("process_files", "aggregate_content")
main_workflow.add_edge("aggregate_content", "generate_answer")
main_workflow.add_edge("generate_answer", END)

app = main_workflow.compile()
