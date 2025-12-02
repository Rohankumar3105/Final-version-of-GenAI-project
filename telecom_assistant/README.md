# Telecom Service Assistant

A comprehensive AI-powered telecom service assistant built with multiple AI frameworks and LangGraph orchestration.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **OpenAI API Key** - Get one from [OpenAI Platform](https://platform.openai.com/)

### 1. Setup Environment

Create a `.env` file in the **parent directory** (`Final version of GenAI project/.env`):

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Install Dependencies

```bash
cd telecom_assistant
pip install -r requirements.txt
```

This will install:
- Streamlit (UI framework)
- LangChain & LangGraph (AI orchestration)
- OpenAI (LLM provider)
- python-dotenv (environment variables)

### 3. Test Database Connection

Verify the database and see sample customer IDs:

```bash
python test_database.py
```

### 4. Test LangGraph Workflow

Test the AI query classification and routing:

```bash
# Quick test (single query)
python quick_test.py

# Full test suite (multiple queries)
python test_graph.py
```

### 5. Run the Application

```bash
streamlit run ui/streamlit_app.py
```

Or use the startup script (Windows):
```bash
run_app.bat
```

The app will open at `http://localhost:8501`

## 🔐 Authentication

### Customer Login
Use any valid Customer ID from the database:
- `CUST001` - SivaPrasad Valluru
- `CUST002` - Rishik V
- `CUST003` - Suresh Patel
- `CUST004` - Ananya Singh
- `CUST005` - Vikram Reddy

### Admin Login
- Enter `admin` as the Customer ID

## ✨ Features

### ✅ Implemented

#### 1. **Authentication System**
- Customer login using Customer ID
- Admin access with "admin" keyword
- Session management with full customer details

#### 2. **LangGraph Orchestration**
- AI-powered query classification using GPT-4
- Intelligent routing to appropriate agent nodes
- Four query categories:
  - 📱 **Billing & Account** → CrewAI Node (mock)
  - 📡 **Network Troubleshooting** → AutoGen Node (mock)
  - 🎯 **Service Recommendations** → LangChain Node (mock)
  - 📚 **Technical Support** → LlamaIndex Node (mock)

#### 3. **Chat Assistant UI**
- Real-time chat interface
- Message history preservation
- AI-powered responses
- Clear chat functionality

### 🚧 Coming Soon

- CrewAI Billing Agents (Billing Specialist + Service Advisor)
- AutoGen Network Troubleshooting (Multi-agent system)
- LangChain Service Recommendations (ReAct agent)
- LlamaIndex Knowledge Retrieval (Vector store + SQL)
- Account Dashboard
- Network Status Monitoring

## 🏗️ Architecture

### Query Flow

```
User Query
    ↓
LangGraph Orchestrator
    ↓
AI Classification (GPT-4)
    ↓
    ├─→ Billing Query → CrewAI Node
    ├─→ Network Issue → AutoGen Node  
    ├─→ Plan Recommendation → LangChain Node
    └─→ Technical Question → LlamaIndex Node
    ↓
Response Formulation
    ↓
User Interface
```

### Project Structure

```
telecom_assistant/
├── app.py                          # Main entry point
├── requirements.txt                # Dependencies
├── test_database.py               # Database testing
├── test_graph.py                  # Full graph testing
├── quick_test.py                  # Quick graph test
├── run_app.bat                    # Windows startup script
│
├── config/
│   └── config.py                  # ✅ Configuration & API keys
│
├── orchestration/
│   ├── state.py                   # ✅ State management
│   └── graph.py                   # ✅ LangGraph workflow
│
├── ui/
│   └── streamlit_app.py           # ✅ Chat UI implemented
│
├── utils/
│   └── database.py                # ✅ Authentication logic
│
├── agents/                        # 🚧 To be implemented
│   ├── billing_agents.py          # CrewAI
│   ├── network_agents.py          # AutoGen
│   ├── service_agents.py          # LangChain
│   └── knowledge_agents.py        # LlamaIndex
│
└── data/
    ├── telecom.db                 # SQLite database
    └── documents/                 # Knowledge base
```

## 🧪 Testing

### Test the Database
```bash
python test_database.py
```
Shows available customer IDs and database status.

### Test LangGraph (Quick)
```bash
python quick_test.py
```
Tests a single billing query through the workflow.

### Test LangGraph (Full)
```bash
python test_graph.py
```
Tests multiple queries across all categories.

### Example Test Queries

**Billing:**
- "Why is my bill higher this month?"
- "What charges are on my account?"

**Network:**
- "I can't make calls"
- "My data is very slow"

**Plans:**
- "Best plan for a family?"
- "Which plan has international roaming?"

**Technical:**
- "How to enable VoLTE?"
- "What are APN settings?"

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Database**: SQLite
- **Orchestration**: LangGraph ✅
- **LLM**: OpenAI GPT-4 ✅
- **AI Frameworks** (pending):
  - CrewAI (Collaborative agents)
  - AutoGen (Multi-agent conversations)
  - LangChain (ReAct agents)
  - LlamaIndex (Knowledge retrieval)

## � Configuration

Edit `config/config.py` to change:
- LLM model (GPT-4 vs GPT-3.5-turbo)
- Temperature settings
- Database paths
- Document storage locations

## 📝 Development Notes

### Current Status
- ✅ Authentication system complete
- ✅ LangGraph orchestration complete
- ✅ Chat UI complete
- ✅ AI-powered query classification
- 🚧 Agent implementations pending (currently return mock responses)

### Next Steps
1. Implement CrewAI billing agents
2. Implement AutoGen network agents
3. Implement LangChain service agents
4. Implement LlamaIndex knowledge agents
5. Add account dashboard
6. Add network monitoring

## 📄 License

Educational project for learning AI agent orchestration.
