# # CrewAI implementation
# import sys
# from pathlib import Path
# from typing import Dict, Any

# # Add parent directory to path
# sys.path.append(str(Path(__file__).parent.parent))

# from crewai import Agent, Task, Crew, Process
# from langchain_community.utilities import SQLDatabase
# from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain_openai import ChatOpenAI
# from config.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, DB_PATH


# # Initialize LLM for agents
# llm = ChatOpenAI(
#     model=LLM_MODEL,
#     temperature=LLM_TEMPERATURE,
#     api_key=OPENAI_API_KEY
# )

# # Initialize database connection
# db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# # Create SQL toolkit for agents
# toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# sql_tools = toolkit.get_tools()


# def create_billing_specialist_agent():
#     """
#     Create a Billing Specialist agent that analyzes bills and charges.
#     """
#     return Agent(
#         role="Billing Specialist",
#         goal="Analyze customer bills, identify charges, and explain billing details clearly",
#         backstory="""You are an experienced telecom billing analyst with 10 years of experience.
#         You excel at breaking down complex bills into understandable components and identifying
#         any unusual charges or billing patterns. You have deep knowledge of telecom billing
#         systems, plan structures, and usage-based charging.""",
#         verbose=True,
#         allow_delegation=False,
#         llm=llm,
#         tools=sql_tools
#     )


# def create_service_advisor_agent():
#     """
#     Create a Service Advisor agent that reviews plans and suggests optimizations.
#     """
#     return Agent(
#         role="Service Advisor",
#         goal="Review customer plans, analyze usage patterns, and provide recommendations for cost optimization",
#         backstory="""You are a telecom service advisor who helps customers get the most value
#         from their plans. You understand all available service plans, add-ons, and promotional
#         offers. You can identify when a customer is on a suboptimal plan and recommend better
#         alternatives based on their usage patterns.""",
#         verbose=True,
#         allow_delegation=False,
#         llm=llm,
#         tools=sql_tools
#     )


# def process_billing_query(customer_id: str, query: str, customer_info: Dict[str, Any] = None) -> str:
#     """
#     Process a billing query using CrewAI agents.
    
#     Args:
#         customer_id: The customer's ID
#         query: The billing question/query
#         customer_info: Optional customer information dictionary
        
#     Returns:
#         A formatted response from the billing crew
#     """
#     try:
#         # Create agents
#         billing_specialist = create_billing_specialist_agent()
#         service_advisor = create_service_advisor_agent()
        
#         # Get customer name for personalization
#         customer_name = customer_info.get('name', 'Valued Customer') if customer_info else 'Valued Customer'
#         plan_id = customer_info.get('plan_id', 'Unknown') if customer_info else 'Unknown'
        
#         # Create Task 1: Analyze billing and charges
#         billing_analysis_task = Task(
#             description=f"""Analyze the billing information for customer {customer_id}.
            
# Customer Query: {query}
# Customer Name: {customer_name}
# Current Plan ID: {plan_id}

# Your tasks:
# 1. Query the customer_usage table to get recent billing information for customer_id = '{customer_id}'
# 2. Query the service_plans table to understand their current plan pricing and features
# 3. Compare current and previous billing periods if applicable
# 4. Identify any additional charges or unusual patterns
# 5. Provide a clear explanation of all charges

# Important database tables to use:
# - customer_usage: Contains billing_period_start, billing_period_end, data_used_gb, voice_minutes_used, 
#   sms_count_used, additional_charges, total_bill_amount
# - service_plans: Contains plan details, monthly_cost, data_limit_gb, voice_minutes, etc.
# - customers: Contains customer information and service_plan_id

# Focus on answering the specific question asked while providing relevant billing context.""",
#             agent=billing_specialist,
#             expected_output="A detailed billing analysis explaining charges and identifying any changes or unusual patterns"
#         )
        
#         # Create Task 2: Review plan and usage optimization
#         plan_review_task = Task(
#             description=f"""Review the customer's plan and usage to provide recommendations.
            
# Customer ID: {customer_id}
# Customer Name: {customer_name}
# Current Plan ID: {plan_id}
# Query: {query}

# Your tasks:
# 1. Analyze the customer's usage patterns from customer_usage table
# 2. Compare their usage with their current plan limits from service_plans table
# 3. Identify if they are underutilizing or exceeding plan limits
# 4. Check if there are better plan options available
# 5. Provide cost optimization recommendations if applicable

# Consider:
# - Are they paying for unused features?
# - Are they incurring additional charges due to plan limitations?
# - Would a different plan save them money or provide better value?

# Provide actionable recommendations only if relevant to their query.""",
#             agent=service_advisor,
#             expected_output="A plan usage analysis with cost optimization recommendations if applicable"
#         )
        
#         # Create Task 3: Generate final response
#         final_response_task = Task(
#             description=f"""Create a comprehensive, customer-friendly response to this billing query.
            
# Customer Query: {query}
# Customer Name: {customer_name}

# Combine the insights from both the Billing Specialist and Service Advisor to create
# a single, well-structured response that:

# 1. Directly answers the customer's question
# 2. Provides specific numbers and details from their account
# 3. Explains any charges or billing changes clearly
# 4. Includes relevant recommendations if applicable
# 5. Uses a friendly, professional tone

# Format the response as a SINGLE PARAGRAPH that flows naturally. Do not use bullet points
# or numbered lists in the final output. Make it conversational and easy to understand.

# Start with addressing the customer by name if appropriate, then provide the answer.""",
#             agent=billing_specialist,
#             expected_output="A single, comprehensive paragraph explaining the billing situation and providing recommendations"
#         )
        
#         # Create the crew
#         crew = Crew(
#             agents=[billing_specialist, service_advisor],
#             tasks=[billing_analysis_task, plan_review_task, final_response_task],
#             process=Process.sequential,
#             verbose=True
#         )
        
#         # Execute the crew
#         result = crew.kickoff()
        
#         # Extract and return the response
#         if result:
#             return str(result)
#         else:
#             return f"I apologize, {customer_name}, but I couldn't process your billing query at this time. Please try again or contact customer support for assistance."
            
#     except Exception as e:
#         print(f"Error in billing crew: {e}")
#         error_response = f"I apologize, but I encountered an error while analyzing your billing information. "
        
#         # Provide helpful fallback based on query type
#         if any(word in query.lower() for word in ['higher', 'increase', 'more', 'expensive']):
#             error_response += "Billing increases can be due to plan changes, additional services, or usage overages. Please contact our billing department at billing@telecom.com for a detailed review of your account."
#         elif any(word in query.lower() for word in ['charge', 'fee', 'cost']):
#             error_response += "For detailed information about specific charges on your bill, please contact our billing department at billing@telecom.com or check your detailed bill in the customer portal."
#         elif any(word in query.lower() for word in ['add-on', 'addon', 'pack', 'service']):
#             error_response += "To learn about available add-on packs and services, please visit our website or contact customer support at support@telecom.com."
#         else:
#             error_response += "Please try rephrasing your question or contact customer support at support@telecom.com for personalized assistance."
        
#         return error_response


"""
Billing agents and tools for the Telecom Assistant using CrewAI.
This module provides:
- BillingDatabaseTool (CrewAI BaseTool)
- get_billing_database_tool()
- create_billing_agents()
- process_billing_query() for integration with LangGraph
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Any

from pydantic import Field
from crewai.tools import BaseTool
from crewai import Agent

# Resolve path to telecom.db relative to project root
BASE_DIR = Path(__file__).parent.parent  # telecom_assistant/
DB_PATH = BASE_DIR / "data" / "telecom.db"


class BillingDatabaseTool(BaseTool):
    """
    CrewAI tool that executes SQL queries and handles simple
    natural-language billing queries against the telecom billing database.
    """

    name: str = "billing_database_tool"
    description: str = (
        "Executes SQL queries and simple natural language queries on the "
        "telecom billing database (data/telecom.db). "
        "Supports queries about bills, charges, usage, and plans."
    )

    db_path: str = Field(default=str(DB_PATH))

    # ---------------------------------------------------------------------
    # Core tool entrypoint used by CrewAI
    # ---------------------------------------------------------------------
    def _run(self, query: str) -> str:
        """
        Handle either:
        - Raw SQL: e.g. 'SELECT COUNT(*) FROM billing'
        - Simple NL: e.g. 'show me recent bills'

        This method is called by CrewAI when the agent chooses this tool.
        """
        q_lower = query.strip().lower()

        # Simple NL → SQL mapping
        nl_map = {
            "show me recent bills": """
                SELECT billing_period, amount, status
                FROM billing
                ORDER BY billing_period DESC
                LIMIT 5;
            """,
            "recent bills": """
                SELECT billing_period, amount, status
                FROM billing
                ORDER BY billing_period DESC
                LIMIT 5;
            """,
        }

        # Choose SQL to execute
        if q_lower in nl_map:
            sql = nl_map[q_lower]
        else:
            # Assume the user / agent passed raw SQL
            sql = query

        return self._execute_sql(sql)

    # ---------------------------------------------------------------------
    # Helper: execute SQL safely and format results
    # ---------------------------------------------------------------------
    def _execute_sql(self, sql: str) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            col_names = [d[0] for d in cur.description] if cur.description else []
            conn.close()
        except Exception as e:
            return f"Database Error: {e}"

        if not rows:
            return "(No results found)"

        # Pretty-print results as table-like text
        if col_names:
            header = " | ".join(col_names)
            sep = "-" * len(header)
            lines = [header, sep]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            return "\n".join(lines)
        else:
            # No column metadata (e.g., result of UPDATE)
            return "\n".join(str(row) for row in rows)


# -------------------------------------------------------------------------
# Factory: return a configured BillingDatabaseTool
# -------------------------------------------------------------------------
def get_billing_database_tool() -> BillingDatabaseTool:
    """
    Returns a fully initialized BillingDatabaseTool instance.
    Used both by tests and by the CrewAI integration.
    """
    return BillingDatabaseTool(db_path=str(DB_PATH))


# -------------------------------------------------------------------------
# Define multiple billing-related agents (for future expansion / testing)
# -------------------------------------------------------------------------
def create_billing_agents() -> Dict[str, Agent]:
    """
    Creates and returns a dictionary of CrewAI Agents that all share
    the same BillingDatabaseTool instance.
    Keys:
        - "billing_expert"
        - "insights_agent"
        - "auditor_agent"
    """
    tool = get_billing_database_tool()

    billing_expert = Agent(
        role="Billing Query Specialist",
        goal=(
            "Explain and resolve customer billing questions using the "
            "telecom billing database."
        ),
        backstory=(
            "You are an expert in telecom billing and account charges. "
            "You can read billing records, usage data, and plan details to "
            "explain why a customer's bill looks the way it does."
        ),
        tools=[tool],
        verbose=True,
    )

    insights_agent = Agent(
        role="Billing Insights Analyst",
        goal=(
            "Analyze billing history and usage patterns to provide insights "
            "and suggestions for optimization."
        ),
        backstory=(
            "You specialize in analyzing historical billing and usage data "
            "to identify trends, anomalies, and potential savings."
        ),
        tools=[tool],
        verbose=True,
    )

    auditor_agent = Agent(
        role="Billing Validation Auditor",
        goal=(
            "Validate billing calculations and detect inconsistencies or "
            "suspicious charges in customer bills."
        ),
        backstory=(
            "You act as a billing auditor who cross-checks charges, taxes, "
            "and discounts to ensure correct billing."
        ),
        tools=[tool],
        verbose=True,
    )

    return {
        "billing_expert": billing_expert,
        "insights_agent": insights_agent,
        "auditor_agent": auditor_agent,
    }


# -------------------------------------------------------------------------
# High-level entrypoint used by LangGraph: process a billing query
# -------------------------------------------------------------------------
from crewai import Task, Crew

def process_billing_query(
    customer_id: str,
    query: str,
    customer_info: Dict[str, Any],
) -> str:
    """
    High-level helper for LangGraph that runs a CrewAI Crew
    consisting of one billing_expert agent + 1 task.
    """

    # Load all agents (tools are already attached)
    agents = create_billing_agents()
    billing_agent = agents["billing_expert"]

    customer_name = customer_info.get("name", customer_id)

    task_prompt = f"""
You are a telecom billing specialist.

Customer ID: {customer_id}
Customer Name: {customer_name}
Additional customer context (may be empty): {customer_info}

Customer question:
{query}

Use the billing_database_tool to:
- Review billing history
- Inspect charges, add-ons, usage, plan costs
- Compare bills with previous months
- Identify reasons for high bills
- Provide a clear explanation in simple terms

Always be friendly and helpful.
"""

    # Create CrewAI Task
    task = Task(
        description=task_prompt,
        agent=billing_agent,
        expected_output="A clear, friendly explanation of the billing issue.",
    )

    # Create Crew
    crew = Crew(
        agents=[billing_agent],
        tasks=[task],
        verbose=True
    )

    # Run the crew
    result = crew.kickoff()

    return str(result)
