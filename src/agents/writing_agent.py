"""
Writing Agent

Specialized agent for generating high-quality written content based on research and analysis.
Handles content generation, formatting, and style adaptation.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
import re

from .base_agent import BaseAgent, AgentMessage


class WritingAgent(BaseAgent):
    """
    Writing Agent specialized in content generation and formatting.
    
    Capabilities:
    - Content generation and synthesis
    - Multi-format output (markdown, HTML, plain text)
    - Style adaptation and tone control
    - Structure optimization
    - Citation and reference management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        capabilities = [
            "content_generation",
            "text_synthesis",
            "format_conversion",
            "style_adaptation",
            "structure_optimization",
            "citation_management"
        ]
        
        super().__init__(
            agent_id="writing_agent",
            name="Writing Agent",
            description="Specialized agent for content generation and formatting",
            capabilities=capabilities,
            config=config
        )
        
        # Writing-specific configuration
        self.default_style = config.get("default_style", "professional") if config else "professional"
        self.max_length = config.get("max_length", 5000) if config else 5000
        self.citation_style = config.get("citation_style", "apa") if config else "apa"
        self.supported_formats = ["markdown", "html", "plain_text", "json"]
    
    def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "writing_request":
            return {
                "status": "acknowledged",
                "task_id": content.get("task_id"),
                "estimated_time": "60-120 seconds"
            }
        elif message_type == "status_query":
            return self.get_metrics()
        else:
            return {"status": "unknown_message_type", "message_type": message_type}
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a writing task"""
        start_time = time.time()
        task_type = task.get("type", "general_writing")
        
        try:
            self.update_state("working", task_type, 0.1)
            
            if task_type == "synthesis_report":
                result = await self._generate_synthesis_report(task)
            elif task_type == "summary":
                result = await self._generate_summary(task)
            elif task_type == "comparative_report":
                result = await self._generate_comparative_report(task)
            elif task_type == "executive_summary":
                result = await self._generate_executive_summary(task)
            elif task_type == "formatted_output":
                result = await self._generate_formatted_output(task)
            else:
                result = await self._general_writing(task)
            
            self.update_state("completed", task_type, 1.0)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, True)
            
            return {
                "status": "success",
                "result": result,
                "processing_time": processing_time,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            self.update_state("error", task_type, 0.0)
            
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, False)
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time": processing_time,
                "agent_id": self.agent_id
            }
    
    async def _generate_synthesis_report(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive synthesis report"""
        research_data = task.get("research_data", {})
        analysis_data = task.get("analysis_data", {})
        style = task.get("style", self.default_style)
        format_type = task.get("format", "markdown")
        
        self.update_state("working", "synthesis_report", 0.2)
        
        # Extract key information
        research_results = research_data.get("results", [])
        analysis_results = analysis_data.get("analysis", {})
        
        # Structure the report
        report_structure = {
            "title": await self._generate_title(research_data, analysis_data),
            "executive_summary": await self._create_executive_summary(research_results, analysis_results),
            "methodology": await self._describe_methodology(research_data, analysis_data),
            "findings": await self._compile_findings(research_results, analysis_results),
            "analysis": await self._synthesize_analysis(analysis_results),
            "conclusions": await self._draw_conclusions(research_results, analysis_results),
            "recommendations": await self._generate_recommendations(analysis_results),
            "references": await self._compile_references(research_results)
        }
        
        self.update_state("working", "synthesis_report", 0.7)
        
        # Generate content for each section
        content_sections = {}
        for section, outline in report_structure.items():
            content_sections[section] = await self._write_section(
                section, outline, style, research_results, analysis_results
            )
        
        self.update_state("working", "synthesis_report", 0.9)
        
        # Format the final report
        formatted_report = await self._format_report(content_sections, format_type)
        
        return {
            "report_type": "synthesis_report",
            "format": format_type,
            "style": style,
            "structure": report_structure,
            "content": formatted_report,
            "metadata": {
                "word_count": self._count_words(formatted_report["full_text"]),
                "section_count": len(content_sections),
                "reference_count": len(report_structure["references"]),
                "generation_timestamp": datetime.now().isoformat()
            }
        }
    
    async def _generate_summary(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a concise summary"""
        data = task.get("data", [])
        summary_type = task.get("summary_type", "general")
        max_length = task.get("max_length", 500)
        format_type = task.get("format", "markdown")
        
        self.update_state("working", "summary", 0.3)
        
        # Extract key points
        key_points = await self._extract_key_points(data, summary_type)
        
        self.update_state("working", "summary", 0.6)
        
        # Generate summary text
        summary_text = await self._create_summary_text(key_points, max_length, summary_type)
        
        # Format summary
        formatted_summary = await self._format_summary(summary_text, format_type)
        
        return {
            "summary_type": summary_type,
            "format": format_type,
            "content": formatted_summary,
            "key_points": key_points,
            "metadata": {
                "word_count": self._count_words(summary_text),
                "compression_ratio": self._calculate_compression_ratio(data, summary_text),
                "key_point_count": len(key_points)
            }
        }
    
    async def _generate_comparative_report(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comparative analysis report"""
        comparison_data = task.get("comparison_data", {})
        format_type = task.get("format", "markdown")
        
        self.update_state("working", "comparative_report", 0.3)
        
        # Structure comparison
        comparison_structure = await self._structure_comparison(comparison_data)
        
        self.update_state("working", "comparative_report", 0.6)
        
        # Generate comparative content
        content = await self._write_comparative_content(comparison_structure)
        
        # Format report
        formatted_report = await self._format_comparative_report(content, format_type)
        
        return {
            "report_type": "comparative_report",
            "format": format_type,
            "content": formatted_report,
            "comparison_structure": comparison_structure,
            "metadata": {
                "comparison_count": len(comparison_data.get("comparisons", {})),
                "dataset_count": comparison_data.get("dataset_count", 0)
            }
        }
    
    async def _generate_executive_summary(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an executive summary"""
        full_report = task.get("full_report", {})
        target_audience = task.get("target_audience", "executives")
        max_length = task.get("max_length", 300)
        
        self.update_state("working", "executive_summary", 0.4)
        
        # Extract executive-level insights
        key_insights = await self._extract_executive_insights(full_report, target_audience)
        
        # Generate concise summary
        exec_summary = await self._write_executive_summary(key_insights, max_length)
        
        return {
            "summary_type": "executive_summary",
            "target_audience": target_audience,
            "content": exec_summary,
            "key_insights": key_insights,
            "metadata": {
                "word_count": self._count_words(exec_summary),
                "insight_count": len(key_insights)
            }
        }
    
    async def _generate_formatted_output(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate output in specific format"""
        content = task.get("content", "")
        source_format = task.get("source_format", "plain_text")
        target_format = task.get("target_format", "markdown")
        
        self.update_state("working", "formatted_output", 0.5)
        
        # Convert format
        converted_content = await self._convert_format(content, source_format, target_format)
        
        return {
            "conversion_type": f"{source_format}_to_{target_format}",
            "source_format": source_format,
            "target_format": target_format,
            "content": converted_content,
            "metadata": {
                "original_length": len(content),
                "converted_length": len(converted_content)
            }
        }
    
    async def _general_writing(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General writing fallback method"""
        return await self._generate_summary(task)
    
    async def _generate_title(self, research_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> str:
        """Generate an appropriate title for the report"""
        query = research_data.get("query", "")
        search_type = research_data.get("search_type", "general")
        
        if query:
            if search_type == "comprehensive_research":
                return f"Comprehensive Analysis: {query.title()}"
            elif search_type == "comparative_analysis":
                return f"Comparative Study: {query.title()}"
            else:
                return f"Research Report: {query.title()}"
        
        return "Multi-Agent RAG System Analysis Report"
    
    async def _create_executive_summary(self, research_results: List[Dict[str, Any]], analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Create executive summary outline"""
        return {
            "purpose": "Overview of research findings and key insights",
            "scope": f"Analysis of {len(research_results)} research sources",
            "key_findings": "Top 3-5 most significant discoveries",
            "recommendations": "Actionable insights and next steps"
        }
    
    async def _describe_methodology(self, research_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Describe the methodology used"""
        return {
            "research_approach": research_data.get("search_type", "semantic search"),
            "analysis_methods": analysis_data.get("analysis_type", "content analysis"),
            "data_sources": f"{len(research_data.get('results', []))} documents analyzed",
            "quality_measures": "Confidence scoring and validation applied"
        }
    
    async def _compile_findings(self, research_results: List[Dict[str, Any]], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile key findings from research and analysis"""
        findings = {
            "research_findings": [],
            "analysis_findings": [],
            "statistical_summary": {}
        }
        
        # Extract research findings
        for result in research_results[:5]:  # Top 5 results
            findings["research_findings"].append({
                "source": result.get("source", "Unknown"),
                "key_content": result.get("content", "")[:200] + "...",
                "relevance_score": result.get("score", 0.0)
            })
        
        # Extract analysis findings
        if "key_themes" in analysis_results:
            findings["analysis_findings"] = analysis_results["key_themes"][:5]
        
        # Statistical summary
        findings["statistical_summary"] = {
            "total_sources": len(research_results),
            "average_relevance": sum(r.get("score", 0) for r in research_results) / len(research_results) if research_results else 0,
            "content_volume": sum(len(r.get("content", "")) for r in research_results)
        }
        
        return findings
    
    async def _synthesize_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize analysis results into coherent insights"""
        synthesis = {
            "thematic_analysis": "",
            "pattern_identification": "",
            "insight_synthesis": ""
        }
        
        # Thematic analysis
        themes = analysis_results.get("key_themes", [])
        if themes:
            top_themes = ", ".join([theme["theme"] for theme in themes[:3]])
            synthesis["thematic_analysis"] = f"Primary themes identified: {top_themes}"
        
        # Pattern identification
        insights = analysis_results.get("insights", [])
        if insights:
            pattern_count = len([i for i in insights if i["type"] in ["repetition_pattern", "content_diversity"]])
            synthesis["pattern_identification"] = f"Identified {pattern_count} significant patterns in the data"
        
        # Insight synthesis
        sentiment = analysis_results.get("sentiment", {})
        if sentiment:
            synthesis["insight_synthesis"] = f"Overall sentiment: {sentiment.get('overall', 'neutral')} (confidence: {sentiment.get('confidence', 0):.2f})"
        
        return synthesis
    
    async def _draw_conclusions(self, research_results: List[Dict[str, Any]], analysis_results: Dict[str, Any]) -> List[str]:
        """Draw conclusions from the combined research and analysis"""
        conclusions = []
        
        # Conclusion from research scope
        if len(research_results) > 10:
            conclusions.append("Comprehensive research base provides robust foundation for analysis")
        elif len(research_results) > 5:
            conclusions.append("Adequate research coverage supports reliable conclusions")
        else:
            conclusions.append("Limited research scope may require additional investigation")
        
        # Conclusion from analysis quality
        themes = analysis_results.get("key_themes", [])
        if themes and len(themes) > 5:
            conclusions.append("Rich thematic diversity indicates comprehensive topic coverage")
        
        # Conclusion from insights
        insights = analysis_results.get("insights", [])
        if insights:
            conclusions.append(f"Generated {len(insights)} actionable insights for further investigation")
        
        return conclusions
    
    async def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on themes
        themes = analysis_results.get("key_themes", [])
        if themes:
            top_theme = themes[0]["theme"]
            recommendations.append(f"Focus deeper investigation on '{top_theme}' as the primary theme")
        
        # Based on insights
        insights = analysis_results.get("insights", [])
        for insight in insights[:2]:  # Top 2 insights
            if insight["confidence"] > 0.7:
                recommendations.append(f"Address {insight['type']}: {insight['insight']}")
        
        # General recommendations
        recommendations.append("Consider expanding research scope for additional perspectives")
        recommendations.append("Validate findings through expert review or additional sources")
        
        return recommendations
    
    async def _compile_references(self, research_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Compile references from research results"""
        references = []
        
        for i, result in enumerate(research_results[:10], 1):  # Limit to 10 references
            metadata = result.get("metadata", {})
            references.append({
                "id": f"ref_{i}",
                "source": result.get("source", f"Source {i}"),
                "title": metadata.get("title", "Untitled Document"),
                "relevance_score": f"{result.get('score', 0.0):.3f}"
            })
        
        return references
    
    async def _write_section(self, section_name: str, outline: Any, style: str, research_results: List[Dict[str, Any]], analysis_results: Dict[str, Any]) -> str:
        """Write content for a specific section"""
        if section_name == "title":
            return outline
        
        elif section_name == "executive_summary":
            return await self._write_executive_summary_content(outline, research_results, analysis_results)
        
        elif section_name == "methodology":
            return await self._write_methodology_content(outline)
        
        elif section_name == "findings":
            return await self._write_findings_content(outline)
        
        elif section_name == "analysis":
            return await self._write_analysis_content(outline)
        
        elif section_name == "conclusions":
            return await self._write_conclusions_content(outline)
        
        elif section_name == "recommendations":
            return await self._write_recommendations_content(outline)
        
        elif section_name == "references":
            return await self._write_references_content(outline)
        
        else:
            return f"Content for {section_name} section"
    
    async def _write_executive_summary_content(self, outline: Dict[str, str], research_results: List[Dict[str, Any]], analysis_results: Dict[str, Any]) -> str:
        """Write the executive summary content"""
        content = []
        
        content.append("## Executive Summary")
        content.append("")
        content.append(f"**Purpose**: {outline['purpose']}")
        content.append(f"**Scope**: {outline['scope']}")
        content.append("")
        
        # Key findings
        content.append("**Key Findings**:")
        themes = analysis_results.get("key_themes", [])
        for i, theme in enumerate(themes[:3], 1):
            content.append(f"{i}. {theme['theme'].title()}: {theme['frequency']} occurrences")
        content.append("")
        
        # Summary of insights
        insights = analysis_results.get("insights", [])
        if insights:
            content.append(f"**Analysis Summary**: Generated {len(insights)} actionable insights from comprehensive analysis.")
            content.append("")
        
        return "\n".join(content)
    
    async def _write_methodology_content(self, outline: Dict[str, str]) -> str:
        """Write the methodology content"""
        content = []
        content.append("## Methodology")
        content.append("")
        content.append(f"**Research Approach**: {outline['research_approach']}")
        content.append(f"**Analysis Methods**: {outline['analysis_methods']}")
        content.append(f"**Data Sources**: {outline['data_sources']}")
        content.append(f"**Quality Measures**: {outline['quality_measures']}")
        content.append("")
        
        return "\n".join(content)
    
    async def _write_findings_content(self, findings: Dict[str, Any]) -> str:
        """Write the findings content"""
        content = []
        content.append("## Research Findings")
        content.append("")
        
        # Research findings
        content.append("### Primary Sources")
        for finding in findings["research_findings"]:
            content.append(f"**Source**: {finding['source']}")
            content.append(f"**Relevance**: {finding['relevance_score']:.3f}")
            content.append(f"**Content**: {finding['key_content']}")
            content.append("")
        
        # Analysis findings
        if findings["analysis_findings"]:
            content.append("### Thematic Analysis")
            for theme in findings["analysis_findings"]:
                content.append(f"- **{theme['theme'].title()}**: {theme['frequency']} occurrences (confidence: {theme.get('confidence', 0):.2f})")
            content.append("")
        
        # Statistical summary
        stats = findings["statistical_summary"]
        content.append("### Statistical Summary")
        content.append(f"- Total Sources: {stats['total_sources']}")
        content.append(f"- Average Relevance: {stats['average_relevance']:.3f}")
        content.append(f"- Content Volume: {stats['content_volume']:,} characters")
        content.append("")
        
        return "\n".join(content)
    
    async def _write_analysis_content(self, synthesis: Dict[str, Any]) -> str:
        """Write the analysis content"""
        content = []
        content.append("## Analysis and Synthesis")
        content.append("")
        content.append(f"**Thematic Analysis**: {synthesis['thematic_analysis']}")
        content.append("")
        content.append(f"**Pattern Identification**: {synthesis['pattern_identification']}")
        content.append("")
        content.append(f"**Insight Synthesis**: {synthesis['insight_synthesis']}")
        content.append("")
        
        return "\n".join(content)
    
    async def _write_conclusions_content(self, conclusions: List[str]) -> str:
        """Write the conclusions content"""
        content = []
        content.append("## Conclusions")
        content.append("")
        
        for i, conclusion in enumerate(conclusions, 1):
            content.append(f"{i}. {conclusion}")
        content.append("")
        
        return "\n".join(content)
    
    async def _write_recommendations_content(self, recommendations: List[str]) -> str:
        """Write the recommendations content"""
        content = []
        content.append("## Recommendations")
        content.append("")
        
        for i, recommendation in enumerate(recommendations, 1):
            content.append(f"{i}. {recommendation}")
        content.append("")
        
        return "\n".join(content)
    
    async def _write_references_content(self, references: List[Dict[str, str]]) -> str:
        """Write the references content"""
        content = []
        content.append("## References")
        content.append("")
        
        for ref in references:
            content.append(f"[{ref['id']}] {ref['title']} - {ref['source']} (Relevance: {ref['relevance_score']})")
        content.append("")
        
        return "\n".join(content)
    
    async def _format_report(self, content_sections: Dict[str, str], format_type: str) -> Dict[str, Any]:
        """Format the complete report"""
        # Combine all sections
        full_text = "\n".join(content_sections.values())
        
        if format_type == "markdown":
            return {
                "full_text": full_text,
                "sections": content_sections,
                "format": "markdown"
            }
        elif format_type == "html":
            html_content = self._convert_markdown_to_html(full_text)
            return {
                "full_text": html_content,
                "sections": {k: self._convert_markdown_to_html(v) for k, v in content_sections.items()},
                "format": "html"
            }
        else:
            plain_text = self._convert_markdown_to_plain(full_text)
            return {
                "full_text": plain_text,
                "sections": {k: self._convert_markdown_to_plain(v) for k, v in content_sections.items()},
                "format": "plain_text"
            }
    
    async def _extract_key_points(self, data: List[Any], summary_type: str) -> List[str]:
        """Extract key points for summary"""
        key_points = []
        
        if summary_type == "research":
            # Extract from research data
            for item in data[:5]:  # Top 5 items
                if isinstance(item, dict):
                    content = item.get("content", "")
                    if content:
                        # Extract first sentence as key point
                        sentences = content.split(". ")
                        if sentences:
                            key_points.append(sentences[0].strip())
        
        elif summary_type == "analysis":
            # Extract from analysis data
            if isinstance(data, dict) and "key_themes" in data:
                for theme in data["key_themes"][:5]:
                    key_points.append(f"{theme['theme'].title()}: {theme['frequency']} occurrences")
        
        else:
            # General extraction
            for item in data[:5]:
                if isinstance(item, str):
                    key_points.append(item[:100] + "..." if len(item) > 100 else item)
                elif isinstance(item, dict):
                    if "content" in item:
                        content = item["content"][:100] + "..." if len(item["content"]) > 100 else item["content"]
                        key_points.append(content)
        
        return key_points
    
    async def _create_summary_text(self, key_points: List[str], max_length: int, summary_type: str) -> str:
        """Create summary text from key points"""
        if not key_points:
            return "No significant information available for summary."
        
        summary_parts = []
        
        if summary_type == "executive":
            summary_parts.append("Executive Summary:")
        elif summary_type == "research":
            summary_parts.append("Research Summary:")
        else:
            summary_parts.append("Summary:")
        
        summary_parts.append("")
        
        current_length = len(" ".join(summary_parts))
        
        for i, point in enumerate(key_points, 1):
            point_text = f"{i}. {point}"
            if current_length + len(point_text) + 2 < max_length:  # +2 for newlines
                summary_parts.append(point_text)
                current_length += len(point_text) + 2
            else:
                break
        
        return "\n".join(summary_parts)
    
    async def _format_summary(self, summary_text: str, format_type: str) -> str:
        """Format summary according to specified format"""
        if format_type == "html":
            return self._convert_markdown_to_html(summary_text)
        elif format_type == "plain_text":
            return self._convert_markdown_to_plain(summary_text)
        else:  # markdown
            return summary_text
    
    def _count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def _calculate_compression_ratio(self, original_data: List[Any], summary_text: str) -> float:
        """Calculate compression ratio"""
        original_length = 0
        for item in original_data:
            if isinstance(item, str):
                original_length += len(item)
            elif isinstance(item, dict):
                original_length += len(str(item))
        
        if original_length == 0:
            return 0.0
        
        return len(summary_text) / original_length
    
    async def _structure_comparison(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure comparison data for report generation"""
        return {
            "datasets": comparison_data.get("dataset_count", 0),
            "criteria": comparison_data.get("comparison_criteria", []),
            "insights": comparison_data.get("insights", [])
        }
    
    async def _write_comparative_content(self, structure: Dict[str, Any]) -> str:
        """Write comparative content"""
        content = []
        content.append("# Comparative Analysis Report")
        content.append("")
        content.append(f"**Datasets Compared**: {structure['datasets']}")
        content.append(f"**Comparison Criteria**: {', '.join(structure['criteria'])}")
        content.append("")
        
        if structure["insights"]:
            content.append("## Key Comparative Insights")
            for insight in structure["insights"]:
                content.append(f"- **{insight['type'].title()}**: {insight['insight']}")
            content.append("")
        
        return "\n".join(content)
    
    async def _format_comparative_report(self, content: str, format_type: str) -> str:
        """Format comparative report"""
        if format_type == "html":
            return self._convert_markdown_to_html(content)
        elif format_type == "plain_text":
            return self._convert_markdown_to_plain(content)
        else:
            return content
    
    async def _extract_executive_insights(self, full_report: Dict[str, Any], target_audience: str) -> List[str]:
        """Extract executive-level insights"""
        insights = []
        
        # Extract high-level findings
        if "analysis" in full_report:
            analysis = full_report["analysis"]
            if "key_themes" in analysis:
                themes = analysis["key_themes"][:3]
                for theme in themes:
                    insights.append(f"Primary focus area: {theme['theme'].title()}")
        
        # Extract strategic implications
        if "recommendations" in full_report:
            recommendations = full_report["recommendations"][:2]
            for rec in recommendations:
                insights.append(f"Strategic recommendation: {rec}")
        
        return insights
    
    async def _write_executive_summary(self, key_insights: List[str], max_length: int) -> str:
        """Write executive summary"""
        content = []
        content.append("# Executive Summary")
        content.append("")
        
        current_length = len("# Executive Summary\n\n")
        
        for insight in key_insights:
            insight_text = f"• {insight}\n"
            if current_length + len(insight_text) < max_length:
                content.append(f"• {insight}")
                current_length += len(insight_text)
            else:
                break
        
        return "\n".join(content)
    
    async def _convert_format(self, content: str, source_format: str, target_format: str) -> str:
        """Convert content between formats"""
        if source_format == target_format:
            return content
        
        if target_format == "html":
            return self._convert_markdown_to_html(content)
        elif target_format == "plain_text":
            return self._convert_markdown_to_plain(content)
        elif target_format == "markdown":
            if source_format == "html":
                return self._convert_html_to_markdown(content)
            else:
                return content
        
        return content
    
    def _convert_markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML"""
        # Simple markdown to HTML conversion
        html = markdown_text
        
        # Headers
        html = re.sub(r'^### (.*$)', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*$)', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.*$)', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
        # Bold
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        
        # Lists
        html = re.sub(r'^\* (.*$)', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^\d+\. (.*$)', r'<li>\1</li>', html, flags=re.MULTILINE)
        
        # Paragraphs
        html = html.replace('\n\n', '</p><p>')
        html = f'<p>{html}</p>'
        
        return html
    
    def _convert_markdown_to_plain(self, markdown_text: str) -> str:
        """Convert markdown to plain text"""
        plain = markdown_text
        
        # Remove markdown formatting
        plain = re.sub(r'^#{1,6} ', '', plain, flags=re.MULTILINE)
        plain = re.sub(r'\*\*(.*?)\*\*', r'\1', plain)
        plain = re.sub(r'\*(.*?)\*', r'\1', plain)
        plain = re.sub(r'^\* ', '• ', plain, flags=re.MULTILINE)
        plain = re.sub(r'^\d+\. ', '', plain, flags=re.MULTILINE)
        
        return plain
    
    def _convert_html_to_markdown(self, html_text: str) -> str:
        """Convert HTML to markdown"""
        # Simple HTML to markdown conversion
        markdown = html_text
        
        # Headers
        markdown = re.sub(r'<h1>(.*?)</h1>', r'# \1', markdown)
        markdown = re.sub(r'<h2>(.*?)</h2>', r'## \1', markdown)
        markdown = re.sub(r'<h3>(.*?)</h3>', r'### \1', markdown)
        
        # Bold
        markdown = re.sub(r'<strong>(.*?)</strong>', r'**\1**', markdown)
        
        # Remove other HTML tags
        markdown = re.sub(r'<[^>]+>', '', markdown)
        
        return markdown
