ROUTING_PROMPT_TEMPLATE = """### Routing Decision Task

**User Request**: 
{user_query}

**Input Context**:
- Type: {input_type}
- Sample: {data_preview}

**Available Pipelines**:
1. `extraction` - For unstructured data parsing
   - Use when: Raw text/JSON/HTML needs structuring
   - Tools: Data extraction, normalization

2. `analysis` - For statistical insights
   - Use when: Need trends, correlations, predictions  
   - Tools: Statistical modeling, visualization

3. `review` - For quality assurance
   - Use when: Validating existing analysis
   - Tools: Accuracy checking, gap analysis

**Output Format** (JSON):
{{
  "selected_crew": "extraction|analysis|review",
  "reason": "Technical justification",
  "parameters": {{
    "detail_level": "basic|standard|detailed",
    "priority": "low|normal|high"
  }}
}}

**Decision Rules**:
- Default to analysis for insight requests
- Prefer extraction for raw data >1000 chars
- Use review only when explicit quality check needed"""

# 示例决策：
EXAMPLE_DECISION = """{
  "selected_crew": "analysis",
  "reason": "Request asks for trend analysis",
  "parameters": {
    "detail_level": "detailed",
    "priority": "normal"
  }
}"""