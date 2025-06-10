from crewai import Crew
from workflow.tasks.analysis_tasks import AnalysisTasks
from workflow.agents.analysis_agents import AnalysisAgents


class ReviewCrew:

    def __init__(self):
        self.agent = AnalysisAgents.create_reviewer()

    def run(self, report_path: str):
        task = AnalysisTasks.review_report(report_path)
        return Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=2
        ).kickoff()



