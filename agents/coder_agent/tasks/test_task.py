from crewai import Agent, Task

class TestTask():
    def test(self, input: str, agent: Agent) -> Task:
        return Task(
            description= f"""Add the testing code in input llm response to the result and append to it the resoponse for the following instruction. 
            Develop a python function to implement test cases for the instruction.
            Mock all inputs including any db connections or file reads and assert.
            import the dev code and necessary functionsfrom generated_code.py
            The description for developer code: {input}
            Create test cases for the above requirements.  
            """,
            goal = "Return a python test function as per user needs",
            expected_output= "Python code snippet",
            agent = agent
        )