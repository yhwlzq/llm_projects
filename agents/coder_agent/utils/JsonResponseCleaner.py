import re
import json

class ResponseCleaner:
    """响应内容清洗工具类（专注提取唯一JSON）"""
    
    @staticmethod
    def clean_response(text: str) -> str:
        """
        从文本中提取第一个有效的JSON数组
        1. 去除所有<think>标签和其他注释
        2. 识别最后一个完整的JSON结构（应对重复情况）pl
        3. 返回格式化后的单一JSON
        """
        # 1. 去除所有<think>和类似标记
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'\[内部思考\].*?\[/内部思考\]', '', text, flags=re.DOTALL)
        
        # 2. 查找所有可能的JSON数组（应对重复）
        json_candidates = re.findall(r'\[.*\]', text, re.DOTALL)
        
        # 3. 逆向检查，返回最后一个有效JSON（通常为最终输出）
        for candidate in reversed(json_candidates):
            try:
                parsed = ast.literal_eval(candidate)  # 比json.loads更安全
                if isinstance(parsed, list):
                    return json.dumps(parsed, indent=2)  # 格式化输出
            except (ValueError, SyntaxError, TypeError):
                continue
        
        # 4. 如果没有JSON，返回清理后的原始文本
        return re.sub(r'\n\s*\n', '\n\n', text).strip()