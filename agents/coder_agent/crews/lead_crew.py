from crewai import Crew

from tasks.lead_task import LeadTask
from agents.lead_agent import LeadAgent
from util.JsonResponseCleaner import ResponseCleaner


class LeadCrew():
    def __init__(self):
        pass

    def run(self, input)->Crew:
        agent = LeadAgent().get_agent()
        tasks = LeadTask().plan(input)
        crew =  Crew(
            tasks=[tasks],
            agents=[agent],
            verbose=True,
        )
        result = crew.kickoff()
        return ResponseCleaner().clean_response(result.raw)