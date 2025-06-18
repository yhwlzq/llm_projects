from crewai import Crew, Process
from tasks.developer_task import DeveloperTask
from agents.developer_agent import DevelopAgent
from util.JsonResponseCleaner import ResponseCleaner
import json


class CodeCrew():
    def __init__(self):
        pass

    def run(self, subtasks)->Crew:
        agent = DevelopAgent().get_agent()
        dev_tasks = []
        print("*****************")
        print(subtasks)
        print("*****************")
        subtasks = json.loads(subtasks)
        for i in subtasks:
            prompt = f"""{i["prompt"]}, Function Name: {i['function_name']}"""
            task = DeveloperTask().develop(prompt,agent)
            dev_tasks.append(task)
        crew =  Crew(
            tasks=[task for task in dev_tasks],
            process=Process.sequential,
            agents=[agent],
            memory=True,
            verbose=True,
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text:latest"
                }
            }
        )
        result = crew.kickoff()
        print(result)
        return ResponseCleaner().clean_response(result.raw)