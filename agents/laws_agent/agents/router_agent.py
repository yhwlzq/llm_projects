from crewai import Agent, Task, Crew
from workflow.prompts.routing_prompt import ROUTING_PROMPT_TEMPLATE
from crewai import Agent
from models.LLMHelper import LLMClient
from typing import Dict, Any
import json



class RouterAgent:

    def __init__(self):
        self.agent = Agent(
            role="Intelligent Workflow Dispatcher",
            goal="Select optimal processing pipeline based on request content",
            backstory="""AI system with deep understanding of data workflows. 
            Uses advanced reasoning to dynamically route tasks.""",
            llm=LLMClient(temperature=0.2),
            verbose=True,
            max_iter=3
        )

    def decide_workflow(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make dynamic routing decision"""
        prompt = self._build_routing_prompt(query, context)
        task = Task(
            description=prompt,
            expected_output="Valid JSON routing decision",
            agent=self.agent,
            output_json=True
        )

        result = Crew(agents=[self.agent], tasks=[task]).kickoff()
        return self._parse_decision(result)

    def _build_routing_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Inject dynamic parameters into prompt template"""
        return ROUTING_PROMPT_TEMPLATE.format(
            user_query=query,
            input_type=context.get('input_type', 'unknown'),
            data_preview=str(context.get('data_sample', ''))[:200]
        )

    def _parse_decision(self, raw_decision: str) -> Dict[str, Any]:
        """Validate and parse routing decision"""
        try:
            decision = json.loads(raw_decision)
            assert decision['selected_crew'] in ['extraction', 'analysis', 'review']
            return decision
        except (json.JSONDecodeError, AssertionError, KeyError):
            return {
                'selected_crew': 'analysis',
                'reason': 'Default fallback',
                'parameters': {'detail_level': 'standard'}
            }