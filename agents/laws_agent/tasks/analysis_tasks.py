from crewai import Task
from typing import List, Dict
from workflow.agents.analysis_agents import AnalysisAgents


class AnalysisTasks:

    @staticmethod
    def route_request(query: str) -> Task:
        """Determine workflow path"""
        return Task(
            description=f"""
            Analyze this request and select the appropriate processing path:

            Request: "{query}"

            Options:
            1. Data Extraction - If request involves parsing raw data
            2. Data Analysis - If request requires statistical insights
            3. Report Review - If evaluating existing analysis

            Respond ONLY with JSON format:
            {{
                "target_agent": "extractor|analyst|reviewer",
                "reason": "brief justification",
                "requires_input": true|false
            }}
            """,
            expected_output="Valid JSON routing decision",
            agent=AnalysisAgents.create_router(),
            output_json=True
        )

    @staticmethod
    def extract_data(rag_data: List[Dict]) -> Task:
        """Data extraction task"""
        return Task(
            description=f"""
            Extract structured data from these RAG results:
            Sample content: {rag_data[0]['content'][:100]}... (total {len(rag_data)} records)

            Requirements:
            - Identify all numerical key-value pairs
            - Convert to clean DataFrame
            - Preserve similarity scores
            """,
            expected_output="Structured pandas DataFrame",
            agent=AnalysisAgents.create_extractor()
        )

    @staticmethod
    def analyze_data(data_frame) -> Task:
        """Data analysis task"""
        return Task(
            description=f"""
            Analyze this dataset (sample):
            {data_frame.head().to_string()}

            Required analyses:
            1. Descriptive statistics
            2. Correlation matrix
            3. Missing values report
            4. Target variable analysis (if 'y' exists)
            """,
            expected_output="Comprehensive analysis dictionary",
            agent=AnalysisAgents.create_analyst()
        )

    @staticmethod
    def generate_report(analysis: Dict) -> Task:
        """Report generation task"""
        return Task(
            description=f"""
            Create professional report from these analysis results:
            Contains keys: {list(analysis.keys())}

            Report must include:
            1. Interactive statistics tables
            2. Correlation visualization
            3. Clear executive summary
            """,
            expected_output="Path to HTML report file",
            agent=AnalysisAgents.create_analyst()
        )

    @staticmethod
    def review_report(report_path: str) -> Task:
        """Quality review task"""
        return Task(
            description=f"""
            Review analysis report at: {report_path}

            Check for:
            1. Mathematical accuracy
            2. Visualization clarity  
            3. Actionable insights
            4. Professional presentation
            """,
            expected_output="Detailed quality assessment report",
            agent=AnalysisAgents.create_reviewer()
        )