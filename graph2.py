from langgraph.graph import StateGraph, END

def add(state):
    return state + 1

def multiply(state):
    return state * 2

def should_continue(state):
    return END if state > 10 else 'multiply'

workflow = StateGraph(int)

workflow.add_node(multiply)
workflow.add_node(add)

workflow.add_edge('multiply', 'add')
workflow.add_conditional_edges('add', should_continue)
workflow.set_entry_point('multiply')

app = workflow.compile()
print(app.invoke(1))