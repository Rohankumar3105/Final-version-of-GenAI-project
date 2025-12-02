"""
AutoGen Network Troubleshooting Agents
---------------------------------------
Latest AutoGen API – NO tools= argument
Tools are now added via agent.register_function(...)
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))

from config.config import OPENAI_API_KEY, LLM_MODEL

from autogen import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from langchain_chroma import Chroma   # NEW correct import for ChromaDB

# Disable Docker for AutoGen
os.environ["AUTOGEN_USE_DOCKER"] = "0"


# ====================================================================
# Create Tools
# ====================================================================
def _create_network_tools():
    """Creates SQL + Vector search tools and returns as dict of callables."""

    BASE_DIR = Path(__file__).parent.parent
    DB_PATH = BASE_DIR / "data" / "telecom.db"
    CHROMA_DIR = BASE_DIR / "data" / "chromadb"

    # -----------------------------
    # SQL Database Tool
    # -----------------------------
    sql_db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    sql_llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    # Create wrapper functions for SQL tools (to avoid method issues)
    def query_sql_database(query: str) -> str:
        """Execute a SQL query against the telecom database."""
        try:
            return sql_db.run(query)
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def get_table_info(table_names: str = "") -> str:
        """Get information about database tables."""
        try:
            return sql_db.get_table_info()
        except Exception as e:
            return f"Error getting table info: {str(e)}"
    
    def list_database_tables() -> str:
        """List all tables in the database."""
        try:
            return ", ".join(sql_db.get_usable_table_names())
        except Exception as e:
            return f"Error listing tables: {str(e)}"

    # -----------------------------
    # Vector Search Tool (Chroma)
    # -----------------------------
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="network_docs",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    def vector_search_tool(query: str) -> str:
        """Search network documentation for relevant information."""
        try:
            results = vector_store.similarity_search(query, k=3)
            return "\n\n".join([str(r.page_content) for r in results])
        except Exception as e:
            return f"Error searching documentation: {str(e)}"

    # Return dictionary of plain functions (not methods)
    return {
        "query_sql_database": query_sql_database,
        "get_table_info": get_table_info,
        "list_database_tables": list_database_tables,
        "vector_search_tool": vector_search_tool,
    }


# ====================================================================
# Create Network Agents
# ====================================================================
def create_network_agents():
    """Creates all AutoGen agents using latest API."""

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    llm_config = {
        "model": LLM_MODEL,
        "temperature": 0,
        "api_key": OPENAI_API_KEY,
        "timeout": 120,
    }

    tools = _create_network_tools()

    # 1. User Proxy Agent
    user_proxy = UserProxyAgent(
        name="user",
        system_message="You represent a telecom customer reporting a network issue.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,
        code_execution_config=False,
    )

    # 2. Network Diagnostics Agent
    network_agent = AssistantAgent(
        name="network_diagnostics",
        llm_config=llm_config,
        system_message="""You are a network diagnostics expert for a telecom company.

CRITICAL: You MUST actually execute SQL queries to check the database. Do not make assumptions.

Your responsibilities:
1. Query the database for network incidents in the customer's area
2. Check cell tower operational status
3. Verify coverage quality

Available functions you MUST use:
- query_sql_database(query) - Execute SQL queries
- list_database_tables() - See available tables
- get_table_info() - Get table schemas

Database tables:
- network_incidents: incident_id, incident_type, location, status, severity, description, start_time, resolution_time
- cell_towers: tower_id, area_id, operational_status, tower_type
- coverage_quality: area_id, technology, signal_strength_category
- service_areas: area_id, city, district, region

Example queries you SHOULD execute:
1. Check incidents: SELECT * FROM network_incidents WHERE location LIKE '%Mumbai%' AND (status = 'In Progress' OR status = 'Active')
2. Check towers: SELECT t.*, s.city, s.district FROM cell_towers t JOIN service_areas s ON t.area_id = s.area_id WHERE s.city LIKE '%Mumbai%'
3. Check coverage: SELECT * FROM coverage_quality WHERE area_id IN (SELECT area_id FROM service_areas WHERE city LIKE '%Mumbai%')

Report your findings with actual data, incident IDs, and tower IDs.
""",
    )

    # Register tools for network agent
    for tool_name, tool_func in tools.items():
        network_agent.register_for_llm(description=tool_func.__doc__ or f"Tool: {tool_name}")(tool_func)
        network_agent.register_for_execution()(tool_func)

    # 3. Device Expert Agent  
    device_agent = AssistantAgent(
        name="device_expert",
        llm_config=llm_config,
        system_message="""You are a device troubleshooting expert.

Provide general troubleshooting steps for common call issues:
1. Basic troubleshooting (airplane mode, restart)
2. Network settings checks
3. SIM card verification
4. Signal strength optimization

If device-specific information is needed, query the device_compatibility table.
Keep advice practical and actionable.
""",
    )

    # Register tools for device agent
    for tool_name, tool_func in tools.items():
        device_agent.register_for_llm(description=tool_func.__doc__ or f"Tool: {tool_name}")(tool_func)
        device_agent.register_for_execution()(tool_func)

    # 4. Solution Integrator Agent
    integrator = AssistantAgent(
        name="solution_integrator",
        llm_config=llm_config,
        system_message="""You are the solution integrator who creates the final troubleshooting report.

WAIT for the network_diagnostics agent to provide SQL query results before creating your report.

Your response MUST include ALL these sections:

SITUATION:
[Write 2-3 complete sentences explaining what's happening. Include specific details from SQL queries like incident IDs, tower IDs, area names, and status. If no incidents were found, state that clearly.]

ROOT CAUSE:
- [List the main cause based on database findings]
- [Add contributing factors if any]
- [Include technical details from the queries]
[If no specific cause found, list: "No active network incidents found in the database for this area"]

RECOMMENDED ACTIONS:
1. [First action based on the situation]
2. [Second action]
3. [Third action]
4. [Fourth action]
5. [Fifth action - include contact support]
[Provide at least 5 specific actions]

ADDITIONAL RECOMMENDATIONS:
- [Device-specific tip]
- [Long-term solution]
- [Prevention advice]
[Provide at least 3 recommendations]

EXPECTED OUTCOME:
[Write 1-2 sentences about what the customer should expect. If there's an outage, mention resolution time. If no outage, mention when to contact support.]

RULES:
- Be specific with data from network_diagnostics agent
- If incident IDs were found, mention them
- If tower IDs were checked, reference them
- If no incidents found, focus on device troubleshooting
- Always provide complete, actionable steps
- Do NOT say "unable to retrieve data" - work with what you have
""",
    )


    # GroupChat
    group = GroupChat(
        agents=[user_proxy, network_agent, device_agent, integrator],
        messages=[],
        max_round=15,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        groupchat=group,
        llm_config=llm_config,
    )

    return user_proxy, manager


# ====================================================================
# Process Network Query
# ====================================================================
def process_network_query(query: str, customer_info: Optional[Dict[str, Any]] = None) -> str:
    """Runs the multi-agent troubleshooting chat and returns final summary.
    
    Args:
        query: The network issue description from the customer
        customer_info: Optional dictionary containing customer details (name, customer_id, etc.)
    
    Returns:
        Formatted troubleshooting response string
    """

    user_proxy, manager = create_network_agents()

    # Enhance query with customer info if provided
    enhanced_query = query
    if customer_info:
        customer_name = customer_info.get("name", "Customer")
        customer_id = customer_info.get("customer_id", "Unknown")
        enhanced_query = f"Customer: {customer_name} (ID: {customer_id})\nIssue: {query}"

    try:
        result = user_proxy.initiate_chat(
            manager,
            message=enhanced_query,
        )

        # Extract final response from chat history
        final_summary = "Unable to generate response."
        
        if hasattr(result, "chat_history") and result.chat_history:
            # Get the last message from solution_integrator or any agent
            for msg in reversed(result.chat_history):
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    # Skip if it's just the user query
                    if content and content != query and content != enhanced_query:
                        final_summary = content
                        break
        
        # Format the response simply
        response = f"""
NETWORK TROUBLESHOOTING REPORT

REPORTED ISSUE:
{query}
"""
        
        if customer_info:
            response += f"""
CUSTOMER INFORMATION:
Name: {customer_info.get('name', 'N/A')}
Customer ID: {customer_info.get('customer_id', 'N/A')}
Phone: {customer_info.get('phone', 'N/A')}
"""
        
        # Add the agent's response
        response += f"""
{final_summary}

---

SUPPORT:
If you need further assistance, please contact our support team with your Customer ID for faster resolution.
"""
        
        return response.strip()
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        return f"""
ERROR IN NETWORK TROUBLESHOOTING

REPORTED ISSUE:
{query}

ERROR DETAILS:
{str(e)}

TROUBLESHOOTING STEPS:
1. Verify your internet connection
2. Check if the database is accessible
3. Ensure all required services are running
4. Contact support if the issue persists

TECHNICAL DETAILS (For Support Team):
{error_details}
"""
