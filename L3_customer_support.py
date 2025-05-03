# L3_customer_support.py (Ollama version with Support Agent, Troubleshooter, and Feedback Reviewer)

from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM

# --- LLM SETUP ---
llm = OllamaLLM(model="llama3")

# --- AGENTS ---

support_agent = Agent(
    role="Customer Support Agent",
    goal="Assist customers by understanding their queries and routing them appropriately",
    backstory="You're a frontline support agent helping customers resolve their issues efficiently by understanding their needs and providing appropriate guidance or escalation.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

troubleshooter = Agent(
    role="Technical Troubleshooter",
    goal="Diagnose and resolve technical issues faced by customers",
    backstory="You specialize in identifying technical problems customers report and providing clear, step-by-step resolutions using your domain expertise.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

feedback_reviewer = Agent(
    role="Customer Feedback Reviewer",
    goal="Analyze customer feedback to identify improvement areas",
    backstory="You are responsible for reviewing customer feedback logs to uncover pain points, patterns, and actionable improvement areas for the support process.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# --- TASKS ---

identify_issue = Task(
    description="Interpret customer query to determine the nature of the support needed, categorize the issue, and decide whether to escalate.",
    expected_output="A categorized issue summary with escalation status.",
    agent=support_agent,
)

diagnose_problem = Task(
    description="Investigate the issue and provide a detailed resolution based on known issues, logs, and troubleshooting steps.",
    expected_output="A technical summary with resolution steps or next actions.",
    agent=troubleshooter,
)

analyze_feedback = Task(
    description="Review the customer conversation and identify improvement points in the support process.",
    expected_output="A list of support improvements and potential training recommendations.",
    agent=feedback_reviewer,
)

# --- CREW SETUP ---

crew = Crew(
    agents=[support_agent, troubleshooter, feedback_reviewer],
    tasks=[identify_issue, diagnose_problem, analyze_feedback],
    verbose=2
)

# --- RUN ---

topic = "Customer is unable to access their account due to a login error"
result = crew.kickoff(inputs={"topic": topic})
print("\n📞 Customer Support Summary:\n", result)
