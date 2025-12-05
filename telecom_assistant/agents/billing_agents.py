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

    return {
        "billing_expert": billing_expert,
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
