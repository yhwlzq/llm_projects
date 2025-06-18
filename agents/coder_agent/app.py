import sys
import logging
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from crews.prompt_optimize_crew import PromptCrew
from crews.code_crew import CodeCrew
from crews.lead_crew import LeadCrew

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment first
load_dotenv()


@dataclass
class PipelineState:
    raw_input: Optional[str] = None
    status_input: Optional[str] = None
    final_output: Optional[str] = None
    error: Optional[Exception] = None

    def validate(self):
        """Validate state transitions"""
        if self.error:
            raise self.error
        if not self.status_input:
            raise ValueError("Missing optimized_prompt before subtasks")
        

class PipelineExecutor(object):

    def __init__(self):
        self.state = PipelineState()
        self.crew_flow = [

             {
                "crew_class": PromptCrew,
                "input_key": "raw_input",      # 从 state.raw_input 获取输入
                "output_key": "optimized_prompt" , # 结果存入 state.optimized_prompt
                "crew_name":"prompt_optimization"
            },
           {
                "crew_class": LeadCrew,
                "input_key": "status_input",
                "output_key": "subtasks",
                 "crew_name":"task_decomposition"
            },
            {
                "crew_class": CodeCrew,
                "input_key": "status_input",
                "output_key": "final_output",
                "crew_name":"code_generation"
            }
        ]
        

    def run_crew(self,crew_class, input_data, crew_name:str):

        try:

            logger.info(f"Starting {crew_name} crew")
            crew = crew_class()
            result = crew.run(input_data)
            logger.info(f"{crew_name} crew completed successfully")
            return result
        
        except Exception as e:
            logger.error(f"{crew_name} crew failed:{e}")
            self.state.error = e
            raise 

    def execute(self,user_input:str):
        previous_output = user_input
        try:
            for config in self.crew_flow:
                crew_class = config['crew_class']
                crew_name = config['output_key']
                result = self.run_crew(crew_class, previous_output, crew_name)
                setattr(self.state, config["output_key"], result)
                previous_output = result
                print(f"\n{crew_name} output:")
                print(result)
                print("---------------------")
            self.state.final_output = previous_output
            return self.state.final_output
        except Exception as e:
            logger.error(f"crew failed:{e}")
            self.state.error = e
            raise 



if __name__ == "__main__":
    
    input_expression  = ""
    while True:
        try:
            input_expression = input("Enter the prompt: ")
            if input_expression.lower() == 'exit':
                sys.exit(0)  
            break  
        except EOFError:
            sys.exit(0)  

    result = PipelineExecutor().execute(input_expression)
    print(result)




