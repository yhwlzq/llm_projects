from crewai import LLM, Agent


class TesterAgent():
    def __init__(self) -> None:
        self.llm = LLM(model='ollama/llama3')

    def get_agent(self) -> Agent:
        return Agent(
            role = "Tester",
            goal = "Write test cases for the python function to implement the instruction and combine all the code snippets into a single python file",
            backstory = "You are a tester that returns a python test cases for all the input function description",
            llm = self.llm,
            verbose= True
        )

    def get_cleaner(self) -> Agent:
        return Agent(
            role = "Tester",
            goal = "Clean the input llm response to remove any unwanted text. Output should be a python code snippet that can be executed",
            backstory = "Support developer that cleans the input llm response and returns a python code snippet that can be executed",
            llm = self.llm,
            verbose= True
        )