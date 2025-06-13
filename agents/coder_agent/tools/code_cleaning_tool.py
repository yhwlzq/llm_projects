import re
from crewai.tools import BaseTool

class CodeCleaningTool(BaseTool):
    name:str = "Code Cleaning Tool"
    description:str = "Useful for cleaning code snippets"

    def _run(self,text:str)->str:
        pattern = r'```(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ''


