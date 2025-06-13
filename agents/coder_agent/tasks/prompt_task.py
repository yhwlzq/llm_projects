from crewai import Task
from typing import Dict, Any
from crewai import Agent

class PromptOptimizerTask:
    """指令优化任务，专门用于将模糊的用户指令转化为清晰可执行的指令"""
    
    def plan(self, user_input: str, agent: Agent) -> Task:
        """
        创建指令优化任务
        
        Args:
            user_input: 需要优化的原始指令
            agent: 负责执行优化的Agent
            
        Returns:
            配置好的优化任务实例
        """
        return Task(
            description=self._build_task_description(user_input),
            agent=agent,
            goal="将模糊的用户指令转化为清晰、具体、可执行的指令",  # 明确的任务目标
            expected_output=self._build_expected_output_description(),

        )
    
    def _build_task_description(self, user_input: str) -> str:
        """构建详细的任务描述"""
        optimization_guidelines = [
            "消除歧义",
            "补充缺失细节",
            "结构化复杂指令",
            "使用明确的行为动词",
            "添加示例说明(如需要)"
        ]
        guidelines = "\n".join(f"- {item}" for item in optimization_guidelines)
        
        return f"""请根据以下指南优化用户指令：
        
            原始指令：
            ----------
            {user_input}
            ----------

            优化指南：
            {guidelines}

            请确保优化后的指令：
            1. 保持原始意图不变
            2. 比原始指令更具体明确
            3. 易于大模型理解执行"""
                
    def _build_expected_output_description(self) -> str:
        """构建预期输出的文本描述"""
        return """请返回一个严格遵循以下格式的JSON对象：
        {
            "optimized_instruction": "优化后的完整指令文本，应该比原始指令更清晰具体",
            "optimization_score": 1-5的整数评分,  // 对优化质量的评分(5分为最佳)
            "original_intent_preserved": true/false  // 是否保持了原始意图
        }

        示例输出:
        {
            "optimized_instruction": "请用Python编写一个完整的贪吃蛇游戏，包含以下功能:1.蛇身移动控制 2.食物生成 3.分数计算 4.游戏结束条件",
            "optimization_score": 4,
            "original_intent_preserved": true
        }"""