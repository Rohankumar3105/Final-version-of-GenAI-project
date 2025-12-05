# Streamlit UI code (Corporate Clean Version)
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path so we can import from other modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.database import authenticate_customer
from utils.document_manager import (
    get_existing_documents,
    save_uploaded_file,
    process_and_index_documents,
    delete_document,
    get_knowledge_base_stats
)

# Lazy-load LangGraph
def get_graph():
    try:
        from orchestration.graph import create_graph
        return create_graph()
    except Exception as e:
        st.error(f"Error loading AI engine: {e}")
        return None

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Telecom Service Assistant",
    layout="wide"
)

# -------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------
defaults = {
    "authenticated": False,
    "user_type": None,
    "customer_id": None,
    "name": None,
    "email": None,
    "phone": None,
    "plan_id": None,
    "chat_history": [],
    "graph": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------------------------------------------
# Sidebar Authentication
# -------------------------------------------------------
with st.sidebar:
    st.title("Telecom Service Assistant")
    st.markdown("---")

    if not st.session_state.authenticated:
        st.subheader("Login")
        customer_id = st.text_input("Customer ID")

        if st.button("Login"):
            if customer_id.strip():
                user_data = authenticate_customer(customer_id.strip())

                if user_data:
                    for key in ["user_type", "customer_id", "name", "email", "phone", "plan_id"]:
                        st.session_state[key] = user_data.get(key)

                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid Customer ID.")
            else:
                st.warning("Please enter a Customer ID.")

        st.markdown("---")
        st.info("If you do not know your Customer ID, please contact customer support.")

    else:
        if st.session_state.user_type == "admin":
            st.success("Admin Access")
        else:
            st.success("Customer Access")

        st.markdown("---")
        st.text(f"Name: {st.session_state.name}")
        st.text(f"Customer ID: {st.session_state.customer_id}")

        if st.session_state.user_type == "customer":
            st.text(f"Email: {st.session_state.email}")
            st.text(f"Phone: {st.session_state.phone}")
            st.text(f"Plan: {st.session_state.plan_id}")

        st.markdown("---")

        if st.button("Logout"):
            for key in defaults:
                st.session_state[key] = defaults[key]
            st.rerun()

# -------------------------------------------------------
# MAIN CONTENT
# -------------------------------------------------------
if st.session_state.authenticated:

    # ---------------------------------------------------
    # CUSTOMER PORTAL
    # ---------------------------------------------------
    if st.session_state.user_type == "customer":
        st.title(f"Welcome, {st.session_state.name}")
        st.markdown("---")

        tab_chat, tab_account, tab_network = st.tabs([
            "Chat Assistant",
            "My Account",
            "Network Status"
        ])

        # ---------------------------------------------------
        # CHAT ASSISTANT (Corporate ChatGPT Style)
        # ---------------------------------------------------
        with tab_chat:
            st.subheader("AI Chat Assistant")
            st.write("You may ask questions regarding billing, plans, network issues, and general support.")

            # Initialize LangGraph
            if st.session_state.graph is None:
                with st.spinner("Initializing AI Assistant..."):
                    st.session_state.graph = get_graph()

            # Display Previous Messages
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # Chat Input
            prompt = st.chat_input("Enter your message")
            if prompt:
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    if st.session_state.graph is None:
                        st.error("AI Assistant failed to initialize.")
                    else:
                        with st.spinner("Processing..."):
                            state = {
                                "query": prompt,
                                "customer_info": {
                                    "customer_id": st.session_state.customer_id,
                                    "name": st.session_state.name,
                                    "email": st.session_state.email,
                                    "phone": st.session_state.phone,
                                    "plan_id": st.session_state.plan_id
                                },
                                "classification": "",
                                "intermediate_responses": {},
                                "final_response": "",
                                "chat_history": st.session_state.chat_history,
                                "error": None
                            }

                            try:
                                result = st.session_state.graph.invoke(state)
                                response = result.get("final_response", "Unable to process your request.")
                                st.write(response)

                                st.session_state.chat_history.append({"role": "assistant", "content": response})

                            except Exception as e:
                                err_message = f"Error processing your request: {e}"
                                st.error(err_message)
                                st.session_state.chat_history.append({"role": "assistant", "content": err_message})

            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

        # ---------------------------------------------------
        # ACCOUNT TAB
        # ---------------------------------------------------
        with tab_account:
            st.subheader("My Account")
            st.write("Account dashboard features will be available soon.")

        # ---------------------------------------------------
        # NETWORK STATUS TAB
        # ---------------------------------------------------
        with tab_network:
            st.subheader("Network Status")
            st.write("Network monitoring features will be available soon.")

    # ---------------------------------------------------
    # ADMIN PORTAL
    # ---------------------------------------------------
    elif st.session_state.user_type == "admin":
        st.title("Administrator Dashboard")
        st.markdown("---")

        tab_kb, tab_support, tab_network, tab_reports = st.tabs([
            "Knowledge Base Management",
            "Customer Support",
            "Network Monitoring",
            "Analytics & Reports"
        ])

        # ---------------------------------------------------
        # KNOWLEDGE BASE MANAGEMENT
        # ---------------------------------------------------
        with tab_kb:
            st.subheader("Knowledge Base Management")

            # Stats
            stats = get_knowledge_base_stats()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Documents", stats["total_documents"])
            c2.metric("Total Size (MB)", stats["total_size_mb"])
            c3.metric("Indexed Chunks", stats["total_chunks"])
            c4.metric("Last Indexed", stats["last_indexed"])

            st.markdown("---")

            # Upload Section
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload files (PDF, MD, TXT)",
                type=["pdf", "md", "txt"],
                accept_multiple_files=True
            )

            if uploaded_files:
                if st.button("Upload"):
                    for file in uploaded_files:
                        save_uploaded_file(file)
                    st.success("Files uploaded successfully.")
                    st.rerun()

            st.markdown("---")

            # Process/Index
            st.subheader("Process and Index Documents")
            if st.button("Process Documents"):
                progress = st.progress(0)
                status = st.empty()

                def update(p, message):
                    progress.progress(p)
                    status.text(message)

                success, message, _ = process_and_index_documents(update)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

            st.markdown("---")

            # Existing Documents
            st.subheader("Existing Documents")
            docs = get_existing_documents()

            if not docs:
                st.info("No documents found.")
            else:
                for i, doc in enumerate(docs):
                    with st.expander(f"{doc['name']}"):
                        st.text(f"Type: {doc['type']}")
                        st.text(f"Size: {doc['size']} bytes")
                        st.text(f"Uploaded: {doc['upload_date']}")

                        if st.button("Delete", key=f"del_{i}"):
                            delete_document(doc["name"])
                            st.success("Document deleted.")
                            st.rerun()

        # ---------------------------------------------------
        # OTHER TABS
        # ---------------------------------------------------
        with tab_support:
            st.subheader("Customer Support")
            st.write("Support dashboard features will be available soon.")

        with tab_network:
            st.subheader("Network Monitoring")
            st.write("Network monitoring features will be available soon.")

        with tab_reports:
            st.subheader("Analytics & Reports")
            st.write("Analytical dashboards will be available soon.")

# -------------------------------------------------------
# LANDING PAGE (IF NOT LOGGED IN)
# -------------------------------------------------------
else:
    st.title("Telecom Service Assistant")
    st.write("Please login using the sidebar to continue.")
