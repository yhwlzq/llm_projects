from crewai import Crew
from workflow.tasks.analysis_tasks import AnalysisTasks
from workflow.agents.analysis_agents import AnalysisAgents

class AnalysisCrew:
    def __init__(self,detail_level="detailed"):
        self.agent = AnalysisAgents.create_extractor()
        self.detail_level = detail_level

    def run(self, data_input):
        if isinstance(data_input, list):  # Raw RAG data
            extract_task = AnalysisTasks.extract_data(data_input)
            analysis_task = AnalysisTasks.analyze_data(extract_task)
        else:  # Already structured data
            analysis_task = AnalysisTasks.analyze_data(data_input)

        report_task = AnalysisTasks.generate_report(
            analysis_task,
            detail_level=self.detail_level
        )

        return Crew(
            agents=[self.agent],
            tasks=[analysis_task, report_task],
            sequential=True,
            verbose=2
        ).kickoff()