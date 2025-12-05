"""
AutoGen Network Troubleshooting Agents
---------------------------------------
Network troubleshooting using:
- UserProxyAgent
- NetworkDiagnosticsAgent
- DeviceExpertAgent
- SolutionIntegratorAgent

Customer-facing output is a simple, corporate-style network
troubleshooting report (Option B1).
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))

from config.config import OPENAI_API_KEY, LLM_MODEL
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from langchain_community.utilities import SQLDatabase

# Base paths
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "telecom.db"

# Ensure AutoGen does not try to use Docker
os.environ["AUTOGEN_USE_DOCKER"] = "0"


# =====================================================================
# Network Tools – Database and Document Search
# =====================================================================
def _create_network_tools():
    """
    Create tools for network diagnosis with proper database and document access.
    
    Tools provided:
    - query_database(query: str) -> Execute SQL queries
    - get_table_schema() -> Get database schema information
    - search_documentation(query: str) -> Search network documentation
    """
    sql_db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    def query_database(query: str) -> str:
        """
        Execute a SQL query against the telecom database.
        Returns the query results or an error message.
        """
        try:
            result = sql_db.run(query)
            if not result or len(result.strip()) == 0:
                return "Query executed but returned no results."
            return result
        except Exception as e:
            return f"SQL Error: {str(e)}"

    def get_table_schema() -> str:
        """
        Get the schema information for all tables in the database.
        This shows available tables and their columns.
        """
        try:
            schema_info = sql_db.get_table_info()
            return schema_info
        except Exception as e:
            return f"Schema Error: {str(e)}"

    def search_documentation(query: str) -> str:
        """
        Search network troubleshooting documentation using vector similarity.
        Returns relevant documentation snippets.
        """
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_chroma import Chroma
            
            CHROMA_DIR = BASE_DIR / "data" / "chromadb"
            
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            vector_store = Chroma(
                collection_name="network_docs",
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
            )
            
            results = vector_store.similarity_search(query, k=3)
            
            if not results:
                return "No relevant documentation found for this query."
            
            doc_text = "\n\n---\n\n".join([doc.page_content for doc in results])
            return f"Found {len(results)} relevant documentation snippets:\n\n{doc_text}"
            
        except Exception as e:
            return f"Documentation Search Error: {str(e)}"

    return {
        "query_database": query_database,
        "get_table_schema": get_table_schema,
        "search_documentation": search_documentation,
    }


# =====================================================================
# Create AutoGen Agents
# =====================================================================
def create_network_agents():
    """Create and return the AutoGen group chat for network troubleshooting."""

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    llm_config = {
        "model": LLM_MODEL,
        "temperature": 0.2,
        "api_key": OPENAI_API_KEY,
        "timeout": 120,
    }

    tools = _create_network_tools()

    # -----------------------------------------------------------------
    # 1. User Proxy Agent – represents the customer
    # -----------------------------------------------------------------
    user_proxy = UserProxyAgent(
        name="user",
        system_message=(
            "You represent a telecom customer who is facing a network issue. "
            "You simply forward the customer's problem to the technical agents. "
            "You do not ask follow-up questions and you do not use tools."
        ),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=4,
    )

    # -----------------------------------------------------------------
    # 2. Network Diagnostics Agent
    # -----------------------------------------------------------------
    network_agent = AssistantAgent(
        name="network_diagnostics",
        llm_config=llm_config,
        system_message="""
You are the Network Diagnostics Specialist with access to the telecom database.

AVAILABLE TOOLS:
1. query_database(query: str) - Execute SQL queries
2. get_table_schema() - Get database schema (use this FIRST if unsure about columns)
3. search_documentation(query: str) - Search troubleshooting documentation

KEY DATABASE TABLES:
- network_incidents: incident_id, incident_type, location, status, severity, description, affected_services
  * location format: "Mumbai Central", "Bangalore South", "Delhi West"
  * status values: "In Progress", "Resolved", "Critical"
  
- service_areas: area_id, city, district, region
  * Links locations to area_ids
  
- cell_towers: tower_id, area_id, operational_status
  * operational_status: "Active", "Maintenance", "Down"
  
- coverage_quality: area_id, technology, signal_strength_category, avg_download_speed_mbps
  * signal_strength_category: "Excellent", "Good", "Fair", "Poor"

- customers: customer_id, name, address, service_plan_id

DIAGNOSTIC WORKFLOW:

STEP 1: IDENTIFY LOCATION
- Extract location from customer issue (city/district like "Mumbai", "Bangalore South")
- If NO location mentioned but customer_info provided, use customer address

STEP 2: CHECK NETWORK INCIDENTS
Query: SELECT incident_id, incident_type, location, status, severity, description 
       FROM network_incidents 
       WHERE location LIKE '%{location}%' AND status IN ('In Progress', 'Critical')

STEP 3: IF NO INCIDENTS, CHECK COVERAGE QUALITY
First get area_id:
  SELECT area_id FROM service_areas WHERE city LIKE '%{location}%' OR district LIKE '%{location}%' LIMIT 1

Then check coverage:
  SELECT signal_strength_category, avg_download_speed_mbps, technology 
  FROM coverage_quality WHERE area_id = '{area_id}'

STEP 4: CHECK CELL TOWERS IF NEEDED
  SELECT t.tower_id, t.operational_status 
  FROM cell_towers t 
  JOIN service_areas s ON t.area_id = s.area_id 
  WHERE s.city LIKE '%{location}%'

RESPONSE FORMAT:

NETWORK_DIAGNOSTICS:
Location: [City/District identified]
Active Incidents: [List incident IDs if found, or "None detected"]
Coverage Status: [Signal quality from coverage_quality table if checked]
Tower Status: [Number of active/down towers if relevant]
Assessment: [Network issue detected / No network issues, likely device-related]

IMPORTANT RULES:
- ALWAYS extract and report location (from query or customer address)
- Only report "In Progress" or "Critical" incidents
- If no incidents found, check coverage_quality table
- Be specific with incident IDs (e.g., "INC003")
- Keep response to 5-6 lines maximum
- If SQL errors, provide best-effort assessment
""",
    )

    # Register network tools
    for name, func in tools.items():
        network_agent.register_for_llm(description=func.__doc__ or name)(func)
        network_agent.register_for_execution()(func)

    # -----------------------------------------------------------------
    # 3. Device Expert Agent
    # -----------------------------------------------------------------
    device_agent = AssistantAgent(
        name="device_expert",
        llm_config=llm_config,
        system_message="""
You are the Device Troubleshooting Expert.

Based on the customer's issue type, provide relevant troubleshooting steps.

ISSUE TYPE DETECTION:
- "can't make calls" / "call drops" / "calling issue" → CALLING PROBLEMS
- "slow internet" / "slow data" / "poor speed" → DATA SPEED PROBLEMS
- "no signal" / "no service" / "no network" → SIGNAL PROBLEMS
- "can't connect" / "connection issue" → CONNECTION PROBLEMS

RESPONSE FORMAT:

DEVICE_TROUBLESHOOTING:
Issue Type: [Calling/Data Speed/Signal/Connection]
Steps:
1. [First specific step for this issue type]
2. [Second specific step]
3. [Third specific step]
4. [Fourth specific step]
5. [Fifth specific step]

CALLING PROBLEMS STEPS:
1. Toggle Airplane Mode ON, wait 10 seconds, then toggle OFF
2. Restart your device completely
3. Go to Settings > Mobile Networks and ensure Voice Calling is enabled
4. Remove your SIM card, clean it gently, and reinsert securely
5. Reset network settings (Settings > System > Reset > Reset Network Settings)

DATA SPEED PROBLEMS STEPS:
1. Turn Mobile Data OFF and ON again
2. Switch to 4G/LTE mode (Settings > Mobile Networks > Preferred Network Type)
3. Clear your browser cache or app cache causing slow speeds
4. Test speed in Safe Mode to rule out third-party app interference
5. Check if you've exceeded your data limit (may cause speed throttling)

SIGNAL PROBLEMS STEPS:
1. Move to an open area away from buildings and test signal
2. Toggle Airplane Mode ON and OFF to re-register with network
3. Manually select network operator (Settings > Mobile Networks > Network Operators)
4. Check if SIM card is properly seated in the tray
5. Test with SIM card in another device to isolate the problem

CONNECTION PROBLEMS STEPS:
1. Restart your device and router/modem (if using WiFi)
2. Forget and reconnect to WiFi network (if WiFi issue)
3. Verify APN settings match your service provider's requirements
4. Disable VPN or proxy settings temporarily
5. Update your device software to the latest version

RULES:
- Choose appropriate steps based on issue type
- Always provide exactly 5 steps
- Keep each step clear and actionable
- No tools needed - just provide the steps
""",
    )

    # (No tools registered for device_agent)

    # -----------------------------------------------------------------
    # 4. Solution Integrator Agent – final customer-facing report
    # -----------------------------------------------------------------
    integrator = AssistantAgent(
        name="solution_integrator",
        llm_config=llm_config,
        system_message="""
You are a friendly chatbot assistant. Create clear, concise, well-formatted responses.

INPUT YOU RECEIVE:
- network_diagnostics message with database findings
- device_expert message with troubleshooting steps

YOUR OUTPUT FORMAT:

---
Hi [Name if available, otherwise skip]!

[One short sentence about their issue with empathy - max 10-15 words]

**Network Status:**
[One clear sentence about what you found. Examples:
- "No outages in Mumbai West - your network is running smoothly."
- "We found an active issue in your area (Incident INC003) - our team is fixing it."
- "Your area has weaker signal right now - let's optimize your device."]

**Quick Fixes:**

Try these steps:

**1. Airplane Mode Reset**
Turn Airplane Mode ON → wait 10 seconds → turn OFF

**2. Restart Device**
Power off completely, then turn back on

**3. Check Settings**
Settings → Mobile Networks → ensure [Voice Calling/Mobile Data] is ON

**4. SIM Card Check**
Remove SIM → clean gently → reinsert firmly

**5. Network Reset**
Settings → System → Reset → Reset Network Settings

**What to expect:**
[Choose ONE concise sentence:
- Network issue: "Service will improve once we fix the network issue (usually 2-4 hours)."
- No issue: "One of these steps should work within 15-30 minutes."
- Poor coverage: "These tweaks should help - consider WiFi calling as backup."]

**Need more help?**
Call us at **198** (24/7)
[If customer ID available: "Customer ID: [ID]"]

---

CRITICAL FORMATTING RULES:
✓ Short paragraphs (1-2 sentences max)
✓ Use bullet points, numbered lists, bold text
✓ NO EMOJIS - keep it clean and professional
✓ Clear visual hierarchy with ** and line breaks
✓ Steps should be scannable - bold the action, then brief explanation
✓ Max 3-4 words for step headers
✓ Keep total response under 200 words
✓ Use → arrow for sequential actions
✓ One blank line between sections

TONE RULES:
- Friendly but professional (not overly casual)
- Direct and clear (no fluff)
- Positive and helpful
- Natural chatbot style (like talking to Google Assistant or Siri)
- Use "you" and "your" but avoid "I" overuse
- One empathy statement max

EXAMPLES OF GOOD STEP FORMATTING:
✓ **Airplane Mode Reset**
   Turn Airplane Mode ON → wait 10 seconds → turn OFF

✓ **Check Mobile Data**
   Settings → Mobile Networks → turn Mobile Data ON

NETWORK STATUS EXAMPLES:
✓ "No outages in Mumbai - network is running smoothly."
✓ "Active network issue detected (Incident INC003) - being resolved."
✓ "Signal is weaker than usual in your area right now."

BE SPECIFIC BUT BRIEF:
- Include incident IDs: "(Incident INC003)"
- Include location: "in Mumbai West"
- Include signal info: "weaker signal" or "good coverage"
- Skip unnecessary explanations
- Get to the point quickly
- NO EMOJIS ANYWHERE IN THE OUTPUT
""",
    )

    # -----------------------------------------------------------------
    # Group Chat Configuration
    # -----------------------------------------------------------------
    group = GroupChat(
        agents=[user_proxy, network_agent, device_agent, integrator],
        messages=[],
        max_round=20,  # Enough rounds for database queries and responses
        allow_repeat_speaker=False,
        speaker_selection_method="round_robin",  # Ensures orderly flow: user -> network -> device -> integrator
    )

    manager = GroupChatManager(groupchat=group, llm_config=llm_config)

    return user_proxy, manager


# =====================================================================
# Public API: Process Network Query
# =====================================================================
def process_network_query(
    query: str,
    customer_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Run the multi-agent network troubleshooting workflow
    and return the final customer-facing report.
    """

    user_proxy, manager = create_network_agents()

    # Build enhanced query with customer context (if available)
    enhanced_query = query
    if customer_info:
        name = customer_info.get("name", "Customer")
        cid = customer_info.get("customer_id", "Unknown")
        enhanced_query = f"Customer: {name} (ID: {cid})\nIssue: {query}"

    try:
        result = user_proxy.initiate_chat(
            manager,
            message=enhanced_query,
        )

        final_report = ""

        # We expect the final answer to come from solution_integrator
        if hasattr(result, "chat_history"):
            for msg in reversed(result.chat_history):
                if isinstance(msg, dict) and msg.get("name") == "solution_integrator":
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content.strip()) > 50:
                        final_report = content.strip()
                        break

        if final_report:
            return final_report

        # Fallback minimal report if integrator output is missing
        return """
NETWORK TROUBLESHOOTING REPORT

SITUATION
We reviewed your reported calling issue.

FINDINGS
We could not generate a complete diagnostic report at this time.

RECOMMENDED ACTIONS
1. Turn Airplane Mode ON and OFF.
2. Restart your device.
3. Ensure mobile data and calling settings are enabled.
4. Remove and reinsert your SIM card.
5. Reset network settings if needed.
6. If the problem continues, contact 198 for support.

EXPECTED OUTCOME
These steps often resolve common calling issues. If the problem persists, our support team can assist with further checks.
""".strip()

    except Exception as e:
        # Hard fallback if AutoGen itself fails
        return f"""
NETWORK TROUBLESHOOTING REPORT

SITUATION
We attempted to analyze your network issue but encountered a system problem.

FINDINGS
The automated diagnostic could not be completed due to a technical error.

RECOMMENDED ACTIONS
1. Turn Airplane Mode ON and OFF.
2. Restart your device.
3. Ensure mobile data and calling settings are enabled.
4. Remove and reinsert your SIM card.
5. Reset network settings if needed.
6. If the problem continues, contact 198 for support.

EXPECTED OUTCOME
These steps typically resolve device-related issues. If the issue persists, our support team will assist you further.
Error reference: {str(e)}
""".strip()
