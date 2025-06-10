import json
from models.LLMHelper import LLMClient
from crews.extraction_crew import ExtractionCrew
from crews.analysis_crew import AnalysisCrew
from crews.review_crew import ReviewCrew
from workflow.agents.router_agent import RouterAgent


class AnalysisWorkFlow:
    def __init__(self):
        self.llm = LLMClient()
        self.router = RouterAgent()
        self.crews = {
            "extraction":ExtractionCrew(),
            "analysis": AnalysisCrew(),
            "review": ReviewCrew(),
        }

    def process(self, user_query: str, input_data=None):
        context = {
            'input_type': type(input_data).__name__,
            'data_sample': input_data[:3] if hasattr(input_data, '__iter__') else input_data
        }
        decision = self.router.decide_workflow(user_query, context)

        print(f"Routing Decision:\n{json.dumps(decision, indent=2)}")
        selected_crew = self.crews[decision['selected_crew']]

        if decision['selected_crew'] == 'analysis':
            selected_crew.set_detail_level(decision['parameters']['detail_level'])

        # 执行工作流
        return selected_crew.run(input_data)
    



