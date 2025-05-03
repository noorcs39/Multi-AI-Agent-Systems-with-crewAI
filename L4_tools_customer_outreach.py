# L4_tools_customer_outreach.py (Ollama version with Tool-Enhanced Outreach Agents)

from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM
from crewai_tools import SerperDevTool, EmailTool

# --- LLM SETUP ---
llm = OllamaLLM(model="llama3")

# --- TOOLS ---
search_tool = SerperDevTool()
email_tool = EmailTool()

# --- AGENTS ---

market_researcher = Agent(
    role="Market Research Specialist",
    goal="Identify high-conversion customer segments and trends",
    backstory="You are responsible for conducting market research using search tools to identify promising outreach targets and tailor strategies accordingly.",
    tools=[search_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

copywriter = Agent(
    role="Sales Copywriter",
    goal="Create engaging and persuasive cold emails for outreach",
    backstory="You specialize in crafting cold email content that resonates with potential clients. You base your work on insights provided by the research team.",
    tools=[],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

outreach_coordinator = Agent(
    role="Outreach Coordinator",
    goal="Send tailored emails and track outreach success",
    backstory="You coordinate personalized email campaigns using tools and templates crafted by the copywriter. You adapt the message to different personas and record metrics.",
    tools=[email_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# --- TASKS ---

research_task = Task(
    description="Use online tools to research the top 3 customer profiles most likely to engage with our product this quarter. Include current pain points, budget size, and where they seek solutions.",
    expected_output="List of 3 customer personas with traits, budgets, and motivation cues.",
    agent=market_researcher,
)

write_emails = Task(
    description="Based on the research findings, write 3 cold email templates targeting the identified personas. Each should be concise, persuasive, and personalized.",
    expected_output="Three cold email templates in markdown format.",
    agent=copywriter,
)

send_outreach = Task(
    description="Use the provided email tool to simulate sending these emails to each persona. Document open rates, replies, or bounces (mock results).",
    expected_output="A delivery and engagement report with email content, delivery status, and interaction summary.",
    agent=outreach_coordinator,
)

# --- CREW SETUP ---

crew = Crew(
    agents=[market_researcher, copywriter, outreach_coordinator],
    tasks=[research_task, write_emails, send_outreach],
    verbose=2
)

# --- RUN ---

topic = "Customer Outreach for B2B SaaS CRM"
result = crew.kickoff(inputs={"topic": topic})
print("\n📬 Outreach Summary:\n", result)
