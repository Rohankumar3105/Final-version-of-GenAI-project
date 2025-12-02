# State management
from typing import TypedDict, Dict, Any, List, Optional


class TelecomAssistantState(TypedDict):
    """
    State structure for the Telecom Assistant workflow.
    This state is passed between all nodes in the LangGraph.
    """
    # User's original query
    query: str
    
    # Customer information from session
    customer_info: Dict[str, Any]
    
    # Query classification result
    classification: str
    
    # Intermediate responses from different agent nodes
    intermediate_responses: Dict[str, Any]
    
    # Final formatted response to user
    final_response: str
    
    # Conversation history
    chat_history: List[Dict[str, str]]
    
    # Optional: Error information if something goes wrong
    error: Optional[str]
