from crewai import Crew
from workflow.tasks.analysis_tasks import AnalysisTasks
from workflow.agents.analysis_agents import AnalysisAgents


class ExtractionCrew:

    def __init__(self):
        self.agent = AnalysisAgents.create_extractor()

    def run(self, rag_data: list):
        task = AnalysisTasks.extract_data(rag_data)
        return Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=2
        ).kickoff()



