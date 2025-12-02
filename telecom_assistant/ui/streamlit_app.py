# Streamlit UI code
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path so we can import from other modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.database import authenticate_customer

# Import LangGraph only when authenticated (lazy loading)
def get_graph():
    """Lazy load the graph to avoid import errors on startup"""
    try:
        from orchestration.graph import create_graph
        return create_graph()
    except Exception as e:
        st.error(f"Error loading LangGraph: {e}")
        return None

# Set page configuration
st.set_page_config(
    page_title="Telecom Service Assistant",
    page_icon="📱",
    layout="wide"
)

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_type" not in st.session_state:
    st.session_state.user_type = None
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None
if "name" not in st.session_state:
    st.session_state.name = None
if "email" not in st.session_state:
    st.session_state.email = None
if "phone" not in st.session_state:
    st.session_state.phone = None
if "plan_id" not in st.session_state:
    st.session_state.plan_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph" not in st.session_state:
    st.session_state.graph = None  # Will be initialized when needed


# Sidebar for authentication
with st.sidebar:
    st.title("📱 Telecom Service Assistant")
    st.markdown("---")
    
    if not st.session_state.authenticated:
        st.subheader("🔐 Login")
        st.markdown("Enter your Customer ID or 'admin' for admin access")
        
        customer_id = st.text_input(
            "Customer ID", 
            placeholder="e.g., CUST001 or admin",
            help="Enter your customer ID or type 'admin' for administrator access"
        )
        
        if st.button("🔑 Login", use_container_width=True):
            if customer_id.strip():
                # Authenticate the user
                user_info = authenticate_customer(customer_id.strip())
                
                if user_info:
                    # Store user information in session state
                    st.session_state.authenticated = True
                    st.session_state.user_type = user_info['user_type']
                    st.session_state.customer_id = user_info['customer_id']
                    st.session_state.name = user_info['name']
                    st.session_state.email = user_info['email']
                    st.session_state.phone = user_info['phone']
                    st.session_state.plan_id = user_info['plan_id']
                    
                    st.success(f"✅ Welcome, {user_info['name']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid Customer ID. Please try again.")
            else:
                st.warning("⚠️ Please enter a Customer ID")
        
        # Help section
        st.markdown("---")
        st.info("💡 **Need Help?**\n\nIf you don't know your Customer ID, please contact support at support@telecom.com")
    
    else:
        # Display user information
        if st.session_state.user_type == "admin":
            st.success("👑 Admin Dashboard")
        else:
            st.success("👤 Customer Portal")
        
        st.markdown("---")
        st.markdown("**User Information:**")
        st.text(f"Name: {st.session_state.name}")
        st.text(f"ID: {st.session_state.customer_id}")
        
        if st.session_state.user_type == "customer":
            st.text(f"Email: {st.session_state.email}")
            st.text(f"Phone: {st.session_state.phone}")
            st.text(f"Plan: {st.session_state.plan_id}")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            # Clear all session state
            st.session_state.authenticated = False
            st.session_state.user_type = None
            st.session_state.customer_id = None
            st.session_state.name = None
            st.session_state.email = None
            st.session_state.phone = None
            st.session_state.plan_id = None
            st.session_state.chat_history = []
            st.success("👋 Logged out successfully!")
            st.rerun()


# Main app content
if st.session_state.authenticated:
    if st.session_state.user_type == "customer":
        # Customer Dashboard
        st.title(f"Welcome, {st.session_state.name}! 👋")
        st.markdown("---")
        
        # Create tabs for different features
        tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📊 My Account", "📡 Network Status"])
        
        with tab1:
            st.header("AI-Powered Chat Assistant")
            st.markdown("Ask me anything about billing, network issues, plans, or technical support!")
            
            # Initialize graph if not already done
            if st.session_state.graph is None:
                with st.spinner("🔧 Initializing AI Assistant..."):
                    st.session_state.graph = get_graph()
                    if st.session_state.graph:
                        st.success("✅ AI Assistant ready!")
                    else:
                        st.error("❌ Failed to initialize AI Assistant. Please check your configuration.")
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("How can I help you today?"):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Process query through LangGraph
                with st.chat_message("assistant"):
                    if st.session_state.graph is None:
                        st.error("AI Assistant not initialized. Please refresh the page.")
                    else:
                        with st.spinner("🤔 Thinking..."):
                            try:
                                # Create state for the graph
                                initial_state = {
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
                                
                                # Run the graph
                                result = st.session_state.graph.invoke(initial_state)
                                response = result.get("final_response", "I apologize, but I couldn't process your request.")
                                
                                # Display response
                                st.markdown(response)
                                
                                # Add assistant response to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": response
                                })
                                
                            except Exception as e:
                                error_msg = f"❌ Error processing your request: {str(e)}"
                                st.error(error_msg)
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": error_msg
                                })
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        with tab2:
            st.header("My Account Information")
            st.info("🚧 Account dashboard coming soon...")
            st.markdown("""
            **Available features will include:**
            - � Usage Statistics
            - 💰 Billing History
            - 📱 Plan Details
            - � Account Settings
            """)
        
        with tab3:
            st.header("Network Status")
            st.info("� Network status monitoring coming soon...")
            st.markdown("""
            **Available features will include:**
            - � Real-time Network Status
            - 🗺️ Coverage Maps
            - ⚠️ Known Issues
            - 🔔 Service Alerts
            """)
        
    elif st.session_state.user_type == "admin":
        # Admin Dashboard
        st.title("Administrator Dashboard 👑")
        st.markdown("---")
        
        # Placeholder for admin features
        st.info("🚧 Admin Dashboard features coming soon...")
        st.markdown("""
        **Available features will include:**
        - 📚 Knowledge Base Management
        - 🎫 Customer Support
        - 📡 Network Monitoring
        - 📊 Analytics & Reports
        - 👥 User Management
        """)

else:
    # Landing page for non-authenticated users
    st.title("Welcome to Telecom Service Assistant 📱")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💬 AI-Powered Support")
        st.write("Get instant help with billing, network issues, and service recommendations")
    
    with col2:
        st.markdown("### 📊 Account Management")
        st.write("View your usage, bills, and manage your telecom services")
    
    with col3:
        st.markdown("### 📡 Network Status")
        st.write("Check real-time network status and coverage in your area")
    
    st.markdown("---")
    st.info("👈 Please login using the sidebar to access all features")
    
    # Feature highlights
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Customers:**
        - 24/7 AI-powered customer support
        - Real-time billing and usage tracking
        - Network troubleshooting assistance
        - Personalized plan recommendations
        - Technical documentation access
        """)
    
    with col2:
        st.markdown("""
        **For Administrators:**
        - Knowledge base management
        - Customer support ticket tracking
        - Network incident monitoring
        - Analytics and reporting
        - System configuration
        """)


# Function to run the app
def main():
    pass  # All logic is above


if __name__ == "__main__":
    main()
