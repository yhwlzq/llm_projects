from dotenv import load_dotenv
import sys
from crews.prompt_optimize_crew import PromptCrew
from crews.lead_crew import LeadCrew
from crews.code_crew import CodeCrew
from tasks.developer_task import DeveloperTask

load_dotenv()

def run():
    input_expression  = ""
    while True:
        try:
            input_expression = input("Enter the prompt: ")
            if input_expression.lower() == 'exit':
                sys.exit(0)  
            break  
        except EOFError:
            sys.exit(0)  

    result = PromptCrew().run(input_expression)

    subtasks = LeadCrew().run(result)

    result = CodeCrew().run(subtasks)

    print("************:\n")

    print(result)

if __name__ == "__main__":
    run()