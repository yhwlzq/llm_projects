from crewai import Agent, LLM
from typing import Optional

class PromptOptimizerAgent:
    """专门用于优化用户指令的Agent，使其表达更清晰、更具体"""
    
    def __init__(self):
        """
        初始化Prompt优化Agent
        
        Args:
            llm: 语言模型实例
        """
        self.llm = LLM(model='ollama/deepseek-r1:latest')
    
    def get_agent(self) -> Agent:
        """
        创建并返回优化指令的Agent
        
        Returns:
            Agent: 配置好的指令优化Agent
        """
        return Agent(
            role="专业指令优化专家",
            goal="优化用户输入的指令，使其表达更清晰、更具体、更易于理解和执行.不要代码，仅仅是指令优化！",
            backstory=(
                "你是一位资深的AI指令优化专家，擅长将模糊、不完整或复杂的用户指令"
                "转化为清晰、具体、可执行的指令。你能识别指令中的歧义点，"
                "并通过提问和补充细节来完善指令。"
            ),
            llm=self.llm,
            verbose=True,
            memory=True,  # 保留优化历史
            max_iter=1,  # 限制优化迭代次数
        )