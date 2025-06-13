from crewai import LLM, Agent

class DevelopAgent():
    def __init__(self):
        self.llm = LLM(model='ollama/deepseek-r1:latest')

    def get_agent(self)->Agent:
        return Agent(
            role='Developer',
            goal = "You are expert of programming, please write Write a python function to implement the instruction and combine all the code snippets into a single python file",
            backstory="you are a developer that returns a python function as per user needs",
            llm = self.llm,
            verbose = True
        )


    def get_clearner(self)->Agent:
        return Agent(
            role='Developer',
            goal="Clean the input llm response to remove any unwanted text. Output should be a python code snippet that can be executed",
            backstory="Support developer that cleans the input llm response and returns a python code snippet that can be executed",
            llm=self.llm,
            verbose=True
        )