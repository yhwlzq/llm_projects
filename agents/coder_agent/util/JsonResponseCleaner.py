import ast
import json
import re


class ResponseCleaner:
    """响应内容清洗工具类（专注提取唯一JSON）"""

    @staticmethod
    def clean_response(text: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'\[内部思考\].*?\[/内部思考\]', '', text, flags=re.DOTALL)
        json_candidates = re.findall(r'\[.*\]', text, re.DOTALL)
        for candidate in reversed(json_candidates):
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, list):
                    return json.dumps(parsed, indent=2)
            except (ValueError, SyntaxError, TypeError):
                continue

        return re.sub(r'\n\s*\n', '\n\n', text).strip()
