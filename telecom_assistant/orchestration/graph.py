# LangGraph Orchestration Layer for Telecom Assistant

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import sys
from pathlib import Path

# Add project root path
sys.path.append(str(Path(__file__).parent.parent))

# Project Imports
from orchestration.state import TelecomAssistantState
from config.config import OPENAI_API_KEY, LLM_MODEL

# Initialize classifier LLM
classifier_llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)


# ============================================================================
# 1. CLASSIFICATION NODE
# ============================================================================
def classify_query(state: TelecomAssistantState) -> TelecomAssistantState:
    """LLM-based classifier that assigns query type."""

    query = state["query"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a telecom assistant query classifier.

Classify the user query into exactly ONE of the following categories:

1. billing_account - Questions about bills, charges, payments, invoices
2. network_troubleshooting - Issues with signal, connectivity, speed, coverage
3. service_recommendation - Requests for plan suggestions, upgrades, data options
4. knowledge_retrieval - Technical questions about settings, features, devices, 5G network coverage areas
5. off_topic - Jokes, greetings, farewells, weather, sports, or any non-telecom query

Examples:

billing_account:
- Why is my bill high?
- What charges were added?

network_troubleshooting:
- My 5G is slow
- No mobile signal

service_recommendation:
- Best plan for roaming?
- I need more data

knowledge_retrieval:
- How to enable VoLTE?
- APN settings?

off_topic:
- Tell me a joke
- Hello / Hi / Good morning
- Thank you / Goodbye
- What's the weather?
- Tell me about sports

Respond ONLY with the category name.
"""),
        ("user", "{query}")
    ])

    try:
        chain = prompt | classifier_llm
        response = chain.invoke({"query": query})
        classification = response.content.strip().lower()

        allowed = [
            "billing_account",
            "network_troubleshooting",
            "service_recommendation",
            "knowledge_retrieval",
            "off_topic",
        ]

        if classification not in allowed:
            classification = "knowledge_retrieval"

        print(f"✅ Query classified as: {classification}")

        return {**state, "classification": classification}

    except Exception as e:
        print(f"❌ Classification error: {e}")
        return {**state, "classification": "knowledge_retrieval"}


# ============================================================================
# 2. ROUTING NODE
# ============================================================================
def route_query(state: TelecomAssistantState) -> str:
    classification = state.get("classification", "")

    mapping = {
        "billing_account": "crew_ai_node",
        "network_troubleshooting": "autogen_node",
        "service_recommendation": "langchain_node",
        "knowledge_retrieval": "llamaindex_node",
        "off_topic": "fallback_handler",
    }

    next_step = mapping.get(classification, "fallback_handler")

    print(f"🔀 Routing to: {next_step}")
    return next_step


# ============================================================================
# 3. CREW AI NODE (Billing)
# ============================================================================
def crew_ai_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """Uses CrewAI to process billing and account-related queries."""

    query = state["query"]
    customer_info = state.get("customer_info", {})
    customer_id = customer_info.get("customer_id", "Unknown")

    print(f"💼 CrewAI Node processing billing query for customer: {customer_id}")

    try:
        from agents.billing_agents import process_billing_query

        response = process_billing_query(
            customer_id=customer_id,
            query=query,
            customer_info=customer_info
        )

        print("✅ CrewAI billing analysis complete")

        return {
            **state,
            "intermediate_responses": {"crew_ai": response}
        }

    except Exception as e:
        print(f"❌ Error in CrewAI node: {e}")

        fail_msg = (
            f"Sorry, we encountered a billing system issue while processing: '{query}'. "
            f"Our billing support team can assist further."
        )

        return {
            **state,
            "intermediate_responses": {"crew_ai": fail_msg},
            "error": str(e)
        }


# ============================================================================
# 4. AUTOGEN NODE (Network Troubleshooting)
# ============================================================================
def autogen_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """
    Executes the AutoGen multi-agent network troubleshooting workflow.
    Returns the final customer-facing report from the solution_integrator agent.
    """

    query = state["query"]
    customer_info = state.get("customer_info", {})

    print("📡 AutoGen Node: Processing network troubleshooting query")

    try:
        # Import AutoGen troubleshooting engine
        from agents.network_agents import process_network_query

        # Run AutoGen multi-agent system with customer context
        final_report = process_network_query(query, customer_info)

        # Validate response
        if not isinstance(final_report, str) or len(final_report.strip()) < 20:
            raise ValueError("Invalid or empty AutoGen response")

        print("✅ AutoGen network troubleshooting completed successfully")

        return {
            **state,
            "intermediate_responses": {"autogen": final_report},
            "error": None,
        }

    except Exception as e:
        print(f"❌ Error in AutoGen node: {e}")

        # Provide helpful fallback response
        fallback = f"""
NETWORK TROUBLESHOOTING ANALYSIS

REPORTED ISSUE:
{query}

SITUATION:
We encountered a technical issue while analyzing your network problem, but we can still provide general troubleshooting guidance.

IMMEDIATE ACTIONS:
1. Toggle Airplane Mode ON, wait 10 seconds, then toggle it OFF
2. Restart your device completely
3. Verify that mobile data is enabled in your device settings
4. Remove and reinsert your SIM card carefully
5. Reset network settings if the issue continues (Settings > Network > Reset)
6. If none of these steps work, call our support team at 198

ADDITIONAL GUIDANCE:
- Check if you're in an area with good network coverage
- Move to a different location and test the connection
- Ensure your device software is up to date
- Try using WiFi calling if available on your plan

EXPECTED OUTCOME:
If the issue is device-related, these steps should resolve it within 15-30 minutes. For persistent problems, our technical team at 198 can provide advanced diagnostics.

SUPPORT CONTACT:
- Customer Service: 198 (available 24/7)
- Reference your Customer ID: {customer_info.get('customer_id', 'N/A')} when calling

Technical Note: {str(e)}
"""

        return {
            **state,
            "intermediate_responses": {"autogen": fallback},
            "error": f"AutoGen error: {str(e)}",
        }


# ============================================================================
# 5. LANGCHAIN NODE (Service Recommendation)
# ============================================================================
def langchain_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """
    Handle service recommendations using the Service Recommendation Agent.
    Calls process_service_query() from service_agents.py
    """

    from agents.service_agents import process_service_query

    query = state["query"]
    customer_info = state.get("customer_info", {})
    customer_id = customer_info.get("customer_id")  # may be None

    print("🎯 LangChain Node: Processing service recommendation query")

    try:
        # Call your real service agent
        response = process_service_query(query, customer_id)

        print("✅ LangChain service recommendation complete")

        return {
            **state,
            "intermediate_responses": {"langchain": response},
            "error": None
        }

    except Exception as e:
        print(f"❌ Error in LangChain node: {e}")

        fallback = f"""
I’m sorry! I couldn’t generate a service recommendation due to an internal error.

Your query was:
{query}

Please try again or contact customer support.
        """

        return {
            **state,
            "intermediate_responses": {"langchain": fallback},
            "error": str(e)
        }


# ============================================================================
# 6. LLAMAINDEX NODE (Knowledge Retrieval)
# ============================================================================
def llamaindex_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """Knowledge search using LlamaIndex Router Engine."""

    query = state["query"]
    print("📚 LlamaIndex Node processing knowledge query")

    try:
        from agents.knowledge_agents import process_knowledge_query

        response = process_knowledge_query(query)

        return {
            **state,
            "intermediate_responses": {"llamaindex": response}
        }

    except Exception as e:
        print(f"❌ LlamaIndex error: {e}")
        return {
            **state,
            "intermediate_responses": {
                "llamaindex": "Error querying technical documentation."
            }
        }


# ============================================================================
# 7. FALLBACK NODE
# ============================================================================
def fallback_handler(state: TelecomAssistantState) -> TelecomAssistantState:
    """Handle off-topic queries with helpful redirection."""
    
    print("⚠️ Fallback Handler Activated")
    
    query = state["query"].lower()
    
    # Detect query type for personalized response
    if any(word in query for word in ["joke", "funny", "laugh", "humor"]):
        response = """
APPRECIATE YOUR INTEREST

I appreciate the lighter moment, but I'm specifically designed to help with telecom services!

Here's what I can assist you with:

BILLING & PAYMENTS:
- View your bills and payment history
- Understand charges and fees
- Set up payment plans
- Check outstanding balances

NETWORK ISSUES:
- Diagnose connectivity problems
- Check coverage in your area
- Troubleshoot slow speeds
- Report network outages

PLAN RECOMMENDATIONS:
- Find the best plan for your needs
- Compare data packages
- International roaming options
- Family and group plans

TECHNICAL SUPPORT:
- Device setup and configuration
- APN settings
- VoLTE and VoWiFi setup
- Feature activation

How can I help you with your telecom needs today?
"""
    
    elif any(word in query for word in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        response = """
HELLO AND WELCOME

Thank you for reaching out! I'm your Telecom Service Assistant, here to help you with all your mobile service needs.

I CAN HELP YOU WITH:

1. BILLING QUESTIONS
   - Review your bill
   - Explain charges
   - Payment assistance

2. NETWORK PROBLEMS
   - Signal issues
   - Slow internet
   - Coverage concerns

3. PLAN RECOMMENDATIONS
   - Find better plans
   - Upgrade options
   - Data packages

4. TECHNICAL SUPPORT
   - Device settings
   - Feature activation
   - Troubleshooting

What would you like assistance with today?
"""
    
    elif any(word in query for word in ["thank", "thanks", "appreciate", "goodbye", "bye", "see you"]):
        response = """
YOU'RE WELCOME

Thank you for using our Telecom Service Assistant! I'm glad I could help.

NEED MORE HELP?

Feel free to ask if you have any other questions about:
- Your bill or account
- Network connectivity
- Service plans
- Technical support

SUPPORT CONTACTS:
- Customer Service: 198
- Technical Support: Available 24/7
- Website: www.telecom.com

Have a great day!
"""
    
    else:
        # Generic off-topic query
        response = """
QUERY NOT RECOGNIZED

I'm specialized in telecom services and may not be able to help with that particular request.

I CAN ASSIST YOU WITH:

BILLING & ACCOUNT MANAGEMENT:
- Bill inquiries and explanations
- Payment plans and history
- Account settings and updates

NETWORK TROUBLESHOOTING:
- Signal and connectivity issues
- Coverage quality checks
- Speed and performance problems

SERVICE RECOMMENDATIONS:
- Plan comparisons and suggestions
- Data package options
- International roaming plans

TECHNICAL DOCUMENTATION:
- Device configuration guides
- Feature setup instructions
- APN and network settings

Please ask me a question related to these telecom services, and I'll be happy to help!

EXAMPLES:
- "Why is my bill higher this month?"
- "I have poor signal at home, can you help?"
- "What's the best plan for heavy data usage?"
- "How do I enable VoLTE on my device?"
"""

    return {**state, "intermediate_responses": {"fallback": response}}


# ============================================================================
# 8. RESPONSE FORMULATION
# ============================================================================
def formulate_response(state: TelecomAssistantState) -> TelecomAssistantState:
    print("📝 Formulating final response")

    responses = state.get("intermediate_responses", {})
    if responses:
        final = next(iter(responses.values()))
    else:
        final = "I couldn't generate a response."

    classification = state.get("classification", "unknown")

    final_response = f"{final}\n\n*Query Type: {classification}*"

    return {**state, "final_response": final_response}


# ============================================================================
# 9. BUILD GRAPH
# ============================================================================
def create_graph():
    print("🔧 Building LangGraph workflow...")

    workflow = StateGraph(TelecomAssistantState)

    # Add nodes
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("crew_ai_node", crew_ai_node)
    workflow.add_node("autogen_node", autogen_node)
    workflow.add_node("langchain_node", langchain_node)
    workflow.add_node("llamaindex_node", llamaindex_node)
    workflow.add_node("fallback_handler", fallback_handler)
    workflow.add_node("formulate_response", formulate_response)

    # Conditional routing
    workflow.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "crew_ai_node": "crew_ai_node",
            "autogen_node": "autogen_node",
            "langchain_node": "langchain_node",
            "llamaindex_node": "llamaindex_node",
            "fallback_handler": "fallback_handler",
        }
    )

    # Output Node
    workflow.add_edge("crew_ai_node", "formulate_response")
    workflow.add_edge("autogen_node", "formulate_response")
    workflow.add_edge("langchain_node", "formulate_response")
    workflow.add_edge("llamaindex_node", "formulate_response")
    workflow.add_edge("fallback_handler", "formulate_response")

    workflow.add_edge("formulate_response", END)

    workflow.set_entry_point("classify_query")

    graph = workflow.compile()

    print("✅ LangGraph workflow created successfully!")
    return graph
