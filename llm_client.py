from typing import Dict, List

from utils import get_openai_client

def generate_response(user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # System prompt: NASA mission operations specialist persona
    system_prompt = (
        "You are a NASA mission operations specialist, highly knowledgeable about the Apollo 11, Apollo 13, and Challenger missions. "
        "You answer questions with technical accuracy, drawing directly from NASA mission transcripts and official documents. "
        "When responding, cite relevant mission details and provide clear, concise, and factual information. "
        "If a question is outside the scope of these missions or the provided context, or if the context does not contain enough information to answer the user's question, explicitly state that you do not have enough information or that the knowledge is missing. "
        "Politely say you can only answer based on the NASA archives and context provided. "
        "Your goal is to assist astronauts, researchers, and historians by delivering trustworthy, well-sourced answers about these historic space missions, and to clearly indicate when information is missing or unavailable."
    )

    print("context for response generation:")
    print(context)
    # User prompt: dynamically include context and user question
    user_prompt = f"""Based on the following context documents, please answer the user's question.\n\nContext Documents:\n{context}\n\nUser Question: {user_message}\n\nIf the context documents are empty or do not contain enough information to answer the question, explicitly state that no relevant information is available or that the knowledge is missing. Otherwise, provide a comprehensive answer based on the context provided."""
    # TODO: Set context in messages
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    # Add conversation history (list of dicts with 'role' and 'content')
    if conversation_history:
        messages.extend(conversation_history)
    # Add the current user prompt last
    messages.append({"role": "user", "content": user_prompt})
    
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return response.choices[0].message.content
