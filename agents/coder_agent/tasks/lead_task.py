from crewai import Agent, Task
from agents.lead_agent import LeadAgent

# class LeadTask:
#     def plan(self, input: str, agent: Agent)-> Task:
#         return Task(
#             description=f"""Split the instruction into smaller tasks, if necessary. Each task should be a prompt for the developer to create a python function.
#                    input: {input}
#                    """,
#             goal="Split the instruction into smaller steps such that for each function there is a prompt",
#             expected_output="""List of prompts for each function needed for the algorithm. Do not include any other text other than the list.
#                    Example: [{'function_name': 'function1', 'prompt': 'prompt1'}, {'function_name': 'function2', 'prompt': 'prompt2'}]""",
#             agent=agent,

#         )



class LeadTask:
    def plan(self, input: str) -> Task:
        agent = LeadAgent().get_agent()
        return Task(
            description=f"""Split the instruction into smaller tasks. Return ONLY a JSON list of function prompts.
                - Input: {input}
                - Rules:
                    1. Output MUST be a list of dictionaries in EXACT format:
                       [{{"function_name": "name", "prompt": "clear prompt"}}]
                    2. DO NOT include ANY other text, explanations, or notes.
                    3. Each prompt must be a direct instruction for a Python function.
                """,
            goal="Generate atomic function prompts in strict JSON format",
            expected_output="""[{"function_name": "func1", "prompt": "..."}, ...]""",
            agent=agent
        )