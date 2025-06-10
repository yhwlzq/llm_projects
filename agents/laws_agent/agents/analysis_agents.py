from crewai import Agent
from models.LLMHelper import LLMClient
from workflow.tools.data_tools import DataAnalysisTools


class AnalysisAgents:

    @staticmethod
    def create_router():
        """Decision maker for task routing"""
        return Agent(
            role="Task Router",
            goal="Determine the best agent for each task",
            backstory="Expert in analyzing requests and assigning to specialists",
            tools=[],
            llm=LLMClient(),
            verbose=True
        )

    @staticmethod
    def create_extractor():
        """Data extraction specialist"""
        return Agent(
            role="Data Extractor",
            goal="Convert unstructured data to structured format",
            backstory="Skilled in parsing and transforming messy data",
            tools=[DataAnalysisTools.extract_from_rag],
            llm=LLMClient(),
            verbose=True
        )

    @staticmethod
    def create_analyst():
        """Data analysis expert"""
        return Agent(
            role="Data Analyst",
            goal="Extract insights from structured data",
            backstory="Statistics expert with keen eye for patterns",
            tools=[DataAnalysisTools.analyze_dataset,DataAnalysisTools.create_report],
            llm=LLMClient(),
            verbose=True
        )

    @staticmethod
    def create_reviewer():
        """Quality assurance specialist"""
        return Agent(
            role="Report Reviewer",
            goal="Ensure analysis quality and clarity",
            backstory="Meticulous professional with high standards",
            tools=[],
            llm=LLMClient(),
            verbose=True
        )