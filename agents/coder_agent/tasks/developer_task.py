from crewai import Task, Agent
from util.JsonResponseCleaner import ResponseCleaner

class DeveloperTask:
    def __init__(self, output_file: str = "combined_code.py"):
        self.output_file = output_file
        # 确保文件存在且是空的
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("# Auto-generated Python code\n\n")

    def _extract_pure_code(self, text: str) -> str:
        """从文本中提取纯Python代码"""
        import re
        # 匹配Python代码块（包含```python或```）
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        # 如果没有代码块，尝试提取看起来像函数/类的部分
        code_lines = []
        in_code = False
        for line in text.split('\n'):
            if line.strip().startswith(('def ', 'class ', '@', 'import ', 'from ')):
                in_code = True
            if in_code:
                code_lines.append(line)
        return '\n'.join(code_lines) if code_lines else text

    def _append_to_file(self, code):
        """将代码追加到文件"""
        try:
            # 获取内容
            if hasattr(code, 'output'):
                content = code.output
            elif hasattr(code, 'result'):
                content = code.result
            elif isinstance(code, str):
                content = code
            else:
                content = str(code)

            content = ResponseCleaner().clean_response(content)
            content = self._extract_pure_code(content)

            # 写入文件
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write("\n\n" + content.strip() + "\n")
                print(f"Successfully wrote to {self.output_file}")
        except Exception as e:
            print(f"Error writing to file: {e}")
            raise

    def develop(self, input: str, agent: Agent) -> Task:
        return Task(
            description=(
                f"Develop a Python function to: {input}\n"
                "IMPORTANT: Return ONLY the complete Python code, "
                "without any explanations, comments or text formatting. "
                "Just the raw executable Python code enclosed in ```python blocks."
            ),
            goal="Generate pure Python code without any additional text",
            expected_output="Only the raw Python code enclosed in ```python ```",
            agent=agent,
            callback=lambda output: self._append_to_file(output)
        )

    def debug(self, dev_code: str, error_msg: str, agent: Agent) -> Task:
        return Task(
            description= f"""
                Dev Code:
                {dev_code}
                \n
                Error Message:
                {error_msg}

                Modify the dev code to fix the error. 
            """,
            goal = "Return modified python code",
            expected_output= "Python code snippet",
            agent= agent
        )