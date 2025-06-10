import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict
from crewai.tools import tool


class DataAnalysisTools:

    @tool("Extract structured data from RAG results")
    def extract_from_rag(rag_results: List[Dict]) -> pd.DataFrame:
        """Convert unstructured RAG content into structured DataFrame"""
        records = []
        for result in rag_results:
            data = {'similarity': result['similarity']}
            for line in result['content'].split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    try:
                        data[key.strip()] = float(val.strip())
                    except ValueError:
                        continue
            records.append(data)
        return pd.DataFrame(records)

    @tool("Perform dataset analysis")
    def analyze_dataset(df: pd.DataFrame) -> Dict:
        """Run comprehensive statistical analysis"""
        analysis = {
            'stats': df.describe().to_dict(),
            'correlations': df.corr().to_dict(),
            'missing': df.isnull().sum().to_dict()
        }
        if 'y' in df.columns:
            analysis['target'] = {
                'mean': df['y'].mean(),
                'range': [df['y'].min(), df['y'].max()],
                'std': df['y'].std()
            }
        return analysis

    @tool("Generate HTML report")
    def create_report(analysis: Dict, output_dir: str = "reports") -> str:
        """Generate visual report with analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "analysis_report.html")

        # Build report content
        html = f"""
        <html><head><title>Analysis Report</title>
        <style>body {{ font-family: Arial; margin: 20px; }}
               .section {{ margin-bottom: 30px; }}
               table {{ border-collapse: collapse; margin-bottom: 20px; }}
               th, td {{ border: 1px solid #ddd; padding: 8px; }}</style>
        </head><body><h1>Data Analysis Report</h1>"""

        # Add stats table
        html += "<div class='section'><h2>Statistics</h2>"
        html += pd.DataFrame(analysis['stats']).to_html()

        # Add correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(analysis['correlations']),
                    annot=True, cmap='coolwarm')
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        html += f"<img src='{heatmap_path}' style='max-width: 100%'>"

        # Add target analysis if exists
        if 'target' in analysis:
            html += "<div class='section'><h2>Target Analysis</h2>"
            html += pd.DataFrame([analysis['target']]).to_html()

        html += "</body></html>"

        with open(report_path, 'w') as f:
            f.write(html)
        return report_path