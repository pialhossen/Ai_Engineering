import dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

dotenv.load_dotenv()

llm = ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0.7,
)

class ContentState(TypedDict):
    topic: str
    draft: str | None
    score: int
    feedback: str
    revision_count: int


def write_draft(state: ContentState):
    prompt = f'''Write a draft on {state['topic']}'''
    if state.get('feedback'):
        prompt += f'\nPrevious feedback to address: {state["feedback"]}'

    response = llm.invoke(prompt)
    text = response.content

    return {
        'draft': text,
        'revision_count': state['revision_count'] + 1
    }

def review_draft(state: ContentState):
    prompt = f'''Rate this paragraph from 1-10 and give one sentence of feedback. be very strict in your feedback.
    Paragraph: {state['draft']}
    Response in exectly this format
    Score: <number>
    Feedback: <text>
    '''
    response = llm.invoke(prompt)
    text = response.content

    score_line = [line for line in text.split('\n') if line.startswith('Score:')][0]
    feedback_line = [line for line in text.split('\n') if line.startswith('Feedback:')][0]

    score = int(score_line.replace('Score:', '').strip())
    feedback = feedback_line.replace('Feedback:', '').strip()

    return {
        'score': score,
        'feedback': feedback
    }

def should_revise(state: ContentState):
    if state['revision_count'] >= 3:
        return END
    if state['score'] >= 8:
        return END
    return 'write_draft'

builder = StateGraph(ContentState)

builder.add_node('write_draft', write_draft)
builder.add_node('review_draft', review_draft)
builder.add_node('should_draft', should_revise)

builder.add_edge(START, 'write_draft')
builder.add_edge('write_draft', 'review_draft')

builder.add_conditional_edges('review_draft', should_revise)
graph = builder.compile()

result = graph.invoke({
    'topic': 'why sleep matters for learning',
    'draft': '',
    'score': 0,
    'feedback': '',
    'revision_count': 0,
})

print(f'Final draft ({result["revision_count"]}) revision(s): \n')
print(result['draft'])
print(f'\n Final Score: {result["score"]}')
