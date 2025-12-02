"""
LangChain Service Recommendation Agent
---------------------------------------
Modern version — NO deprecated agent APIs.
Fully aligned with requirement 1.9.3:
Plan recommendations for generic queries (not customer-specific).
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))

from config.config import OPENAI_API_KEY, LLM_MODEL
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase


# ====================================================================
# Helper Functions
# ====================================================================
def estimate_data_usage(activities: str) -> str:
    """
    Estimate monthly data usage based on user activity description.
    Used for classifying user as light / moderate / heavy / family-heavy.
    """
    try:
        act = activities.lower()
        total = 0
        notes = []

        import re
        hours_match = re.search(r"(\d+)\s*hour", act)
        hours = int(hours_match.group(1)) if hours_match else 2

        # Streaming
        if "video" in act or "stream" in act:
            if "daily" in act:
                gb = hours * 3 * 30
            elif "weekly" in act:
                gb = hours * 3 * 4
            else:
                gb = 60
            total += gb
            notes.append(f"Video streaming: {gb} GB")

        # Work from home
        if "work from home" in act or "remote" in act:
            total += 25
            notes.append("WFH usage: 25 GB")

        # Social media
        if any(x in act for x in ["social", "instagram", "facebook", "whatsapp"]):
            total += 2
            notes.append("Social: 2 GB")

        # Light calling user
        if "call" in act and "video" not in act:
            total += 1

        # Family multiplier
        fam = re.search(r"family of (\d+)", act)
        if fam:
            n = int(fam.group(1))
            total *= n
            notes.append(f"Family multiplier: {n}×")

        total = round(total * 1.2, 1)   # 20% buffer

        # Category
        cat = (
            "Light User (0–10 GB)" if total <= 10 else
            "Moderate User (10–50 GB)" if total <= 50 else
            "Heavy User (50–200 GB)" if total <= 200 else
            "Ultra/Family Heavy User (200+ GB)"
        )

        return (
            f"Estimated Monthly Usage: {total} GB\n"
            f"Category: {cat}\nBreakdown:\n"
            + "\n".join(f"- {n}" for n in notes)
        )

    except Exception:
        return "Unable to estimate usage. Provide more detailed activities."


def calculate_plan_value(plan_row: dict) -> str:
    """
    Evaluates a plan based on cost and data.
    """
    try:
        cost = float(plan_row.get("monthly_cost", 0) or 0)
        data = int(plan_row.get("data_limit_gb", 0) or 0)
        unlimited_data = bool(plan_row.get("unlimited_data"))
        unlimited_voice = bool(plan_row.get("unlimited_voice"))

        if unlimited_data:
            return f"Unlimited data at ₹{cost}/month — excellent for heavy users."

        if data > 0:
            cpg = cost / data
            rating = (
                "Excellent value" if cpg < 100 else
                "Good value" if cpg < 200 else
                "Premium"
            )
            out = f"Cost per GB: ₹{cpg:.2f} — {rating}"
        else:
            out = f"Monthly cost: ₹{cost}"

        if unlimited_voice:
            out += " (Unlimited calling included)"

        return out
    except:
        return "Value analysis unavailable."


# ====================================================================
# Internal helpers: LLM + DB
# ====================================================================
def _create_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )


def _create_sql_db():
    base = Path(__file__).parent.parent
    path = base / "data" / "telecom.db"
    return SQLDatabase.from_uri(f"sqlite:///{path}")


def _fetch_all_plans(sql_db: SQLDatabase):
    """
    Fetch service plans from DB.
    We return text (not dict) because SQLDatabase.run → LLM-friendly string.
    """
    query = """
    SELECT
        plan_id,
        name,
        monthly_cost,
        data_limit_gb,
        unlimited_data,
        unlimited_voice,
        sms_count,
        unlimited_sms,
        international_roaming,
        description
    FROM service_plans;
    """
    return sql_db.run(query)


# ====================================================================
# Main API
# ====================================================================
def process_service_query(query: str, customer_id: Optional[str] = None) -> str:
    """
    The ONLY public function used by LangGraph node.
    Fully aligned with requirement 1.9.3.
    NO need to fetch per-customer data.
    """
    # Check if query is telecom-related first
    if not is_telecom_related(query):
        return handle_fallback_query(query)
    
    llm = _create_llm()
    sql_db = _create_sql_db()

    try:
        plans_text = _fetch_all_plans(sql_db)

        # -----------------------------------
        # Build LLM prompt
        # -----------------------------------
        system_prompt = """
You are a telecom plan recommendation expert.

You MUST recommend the best telecom plans based ONLY on:
- User query
- The list of available service plans (see below)

Your responsibilities:
1. Understand the user's need (e.g., heavy streaming, family plan, WFH, cheap plan)
2. Categorize the user's data usage using reasoning
3. Select 1–3 best plans from service_plans
4. For each recommended plan, provide:
   - Name and plan_id
   - Data allowance (or unlimited_data)
   - Voice/SMS features
   - Monthly cost
   - International roaming availability (if relevant)
   - Value analysis
5. End with a final concise recommendation summary.

IMPORTANT RULES:
- Do NOT create imaginary plans.
- Choose only from the provided service_plans table.
"""

        prompt = (
            f"{system_prompt}\n"
            f"=== USER QUERY ===\n{query}\n\n"
            f"=== AVAILABLE PLANS (service_plans) ===\n{plans_text}\n\n"
            f"Now recommend the best matching plan(s) and explain your reasoning in detail."
        )

        llm_output = llm.invoke(prompt)
        result = llm_output.content if hasattr(llm_output, "content") else str(llm_output)

        # -----------------------------------
        # Final structured response
        # -----------------------------------
        return f"""
📡 **Service Recommendation**

**Your Query:**  
{query}

---

{result}

---

Need help activating a plan?  
Visit our store or contact customer support.
""".strip()

    except Exception as e:
        import traceback
        return f"""
❌ ERROR IN SERVICE RECOMMENDATION

Query: {query}
Error: {e}

Debug:
{traceback.format_exc()}
"""

# ====================================================================
# Fallback Handler for Irrelevant Queries
# ====================================================================
def handle_fallback_query(query: str) -> str:
    """
    Handle queries that are not related to telecom services, billing, or network issues.
    Examples: jokes, greetings, general questions, off-topic conversations
    
    Args:
        query: The user's query
        
    Returns:
        A polite redirect message
    """
    
    query_lower = query.lower()
    
    # Detect query type for appropriate response
    is_joke = any(word in query_lower for word in ["joke", "funny", "humor", "laugh"])
    is_greeting = any(word in query_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"])
    is_thanks = any(word in query_lower for word in ["thank", "thanks", "appreciate"])
    is_goodbye = any(word in query_lower for word in ["bye", "goodbye", "see you", "later"])
    
    # Generate appropriate response
    if is_joke:
        response = f"""
ASSISTANT RESPONSE

YOUR MESSAGE:
{query}

RESPONSE:
I appreciate your sense of humor! While I'd love to share a telecom joke, I'm specifically designed to help you with:

- Service plan recommendations and comparisons
- Network troubleshooting and connectivity issues  
- Billing inquiries and usage questions
- Account management and technical support

How can I assist you with your telecom needs today?

EXAMPLES OF QUESTIONS I CAN HELP WITH:
- "What's the best plan for heavy data usage?"
- "I can't make calls in my area, what should I do?"
- "Why is my bill higher this month?"
- "How do I upgrade my current plan?"
"""
    
    elif is_greeting:
        response = f"""
ASSISTANT RESPONSE

Hello! Welcome to our Telecom Assistant.

I'm here to help you with:

SERVICE RECOMMENDATIONS:
- Find the perfect plan for your needs
- Compare plans based on data, voice, and SMS
- Get recommendations for family or individual use

NETWORK SUPPORT:
- Troubleshoot connectivity issues
- Check network coverage in your area
- Fix call drops and signal problems

BILLING & ACCOUNT:
- Review your usage and charges
- Understand your bill
- Manage your account

How can I assist you today? Please describe your telecom-related question or concern.
"""
    
    elif is_thanks:
        response = f"""
ASSISTANT RESPONSE

You're welcome! I'm glad I could help.

Is there anything else you'd like to know about:
- Service plans and pricing
- Network coverage or connectivity
- Your account or billing

Feel free to ask if you have more questions!
"""
    
    elif is_goodbye:
        response = f"""
ASSISTANT RESPONSE

Thank you for using our Telecom Assistant!

If you need help in the future with:
- Choosing a service plan
- Network troubleshooting
- Billing questions
- Account management

Don't hesitate to reach out. Have a great day!
"""
    
    else:
        # General off-topic query
        response = f"""
ASSISTANT RESPONSE

YOUR MESSAGE:
{query}

RESPONSE:
I'm a specialized Telecom Assistant designed to help with telecom-related services. 
Your query doesn't appear to be related to telecommunications.

I CAN HELP YOU WITH:

1. SERVICE PLANS:
   - Recommend plans based on your usage
   - Compare different plan options
   - Explain plan features and pricing

2. NETWORK ISSUES:
   - Troubleshoot connectivity problems
   - Check coverage in your area
   - Fix call drops and data issues

3. BILLING & ACCOUNT:
   - Review charges and usage
   - Explain billing details
   - Manage account settings

PLEASE ASK ABOUT:
- "Which plan is best for my needs?"
- "Why isn't my network working?"
- "Can you explain my bill?"
- "How do I change my plan?"

How can I help you with your telecom services today?
"""
    
    return response.strip()


# ====================================================================
# Query Classification
# ====================================================================
def is_telecom_related(query: str) -> bool:
    """
    Determine if a query is related to telecom services.
    
    Args:
        query: The user's query
        
    Returns:
        True if telecom-related, False otherwise
    """
    
    query_lower = query.lower()
    
    # Telecom-related keywords
    telecom_keywords = [
        # Service/Plan related
        "plan", "package", "subscription", "service", "upgrade", "downgrade",
        "family plan", "unlimited", "data", "voice", "sms", "international",
        "roaming", "prepaid", "postpaid",
        
        # Network related
        "network", "signal", "coverage", "call", "internet", "wifi", "4g", "5g",
        "connection", "speed", "slow", "fast", "bandwidth", "tower", "outage",
        
        # Billing related
        "bill", "billing", "charge", "payment", "cost", "price", "fee", "usage",
        "invoice", "balance", "due", "overcharge",
        
        # Technical issues
        "not working", "can't make calls", "no signal", "dropped call", 
        "can't connect", "error", "problem", "issue", "troubleshoot",
        
        # Account related
        "account", "customer id", "phone number", "activate", "deactivate",
        "cancel", "renew", "contract"
    ]
    
    # Check if query contains telecom keywords
    has_telecom_keywords = any(keyword in query_lower for keyword in telecom_keywords)
    
    # Exclude obvious non-telecom queries
    non_telecom_keywords = [
        "joke", "funny", "story", "weather", "news", "sports", "movie",
        "recipe", "cooking", "game", "music", "book", "travel"
    ]
    
    has_non_telecom = any(keyword in query_lower for keyword in non_telecom_keywords)
    
    # Greeting/Farewell patterns (these are okay, but not service queries)
    is_greeting_only = query_lower.strip() in [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "bye", "goodbye", "thanks", "thank you"
    ]
    
    # If it's just a greeting, it's not a telecom service query
    if is_greeting_only:
        return False
    
    # If it has non-telecom keywords and no telecom keywords, it's not related
    if has_non_telecom and not has_telecom_keywords:
        return False
    
    # If it has telecom keywords, it's related
    if has_telecom_keywords:
        return True
    
    # Default: if unclear, treat as non-telecom (fallback will handle it)
    return False


