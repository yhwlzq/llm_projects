from crewai import LLM, Agent


class LeadAgent:
    def __init__(self):
        self.llm = LLM(model='ollama/deepseek-r1:latest')

    def get_agent(self)->Agent:
        return Agent(
            role="Software Development Technical Lead",
            goal="Breakdown the user instruction into smaller steps if required, else return the user instruction",
            backstory="You are a tech lead that breakdowns the user instruction into smaller individual tasks if it needs multiple functions",
            llm=self.llm,
            verbose=True
        )