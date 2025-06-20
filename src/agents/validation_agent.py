"""
Validation Agent

Specialized agent for validating results, checking quality, and ensuring accuracy.
Performs quality assurance across the multi-agent workflow.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
import re

from .base_agent import BaseAgent, AgentMessage


class ValidationResult:
    """Represents a validation result"""
    
    def __init__(self, is_valid: bool, confidence: float, issues: List[str], suggestions: List[str]):
        self.is_valid = is_valid
        self.confidence = confidence
        self.issues = issues
        self.suggestions = suggestions
        self.timestamp = datetime.now()


class ValidationAgent(BaseAgent):
    """
    Validation Agent specialized in quality assurance and result verification.
    
    Capabilities:
    - Content quality validation
    - Factual consistency checking
    - Format and structure validation
    - Source credibility assessment
    - Cross-reference verification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        capabilities = [
            "content_validation",
            "quality_assessment",
            "consistency_checking",
            "format_validation",
            "source_verification",
            "accuracy_scoring"
        ]
        
        super().__init__(
            agent_id="validation_agent",
            name="Validation Agent",
            description="Specialized agent for quality assurance and validation",
            capabilities=capabilities,
            config=config
        )
        
        # Validation-specific configuration
        self.quality_threshold = config.get("quality_threshold", 0.7) if config else 0.7
        self.consistency_threshold = config.get("consistency_threshold", 0.8) if config else 0.8
        self.max_issues_reported = config.get("max_issues_reported", 10) if config else 10
        
        # Validation rules and patterns
        self.format_patterns = {
            "markdown": [
                r"^#+ .+",  # Headers
                r"\*\*.+\*\*",  # Bold text
                r"\[.+\]\(.+\)"  # Links
            ],
            "citation": [
                r"\[ref_\d+\]",  # Reference citations
                r"\[\d+\]"  # Numbered citations
            ]
        }
    
    def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "validation_request":
            return {
                "status": "acknowledged",
                "task_id": content.get("task_id"),
                "estimated_time": "30-60 seconds"
            }
        elif message_type == "quality_check":
            return {
                "status": "acknowledged",
                "validation_type": "quality_check"
            }
        elif message_type == "status_query":
            return self.get_metrics()
        else:
            return {"status": "unknown_message_type", "message_type": message_type}
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a validation task"""
        start_time = time.time()
        task_type = task.get("type", "general_validation")
        
        try:
            self.update_state("working", task_type, 0.1)
            
            if task_type == "content_validation":
                result = await self._validate_content(task)
            elif task_type == "workflow_validation":
                result = await self._validate_workflow(task)
            elif task_type == "quality_assessment":
                result = await self._assess_quality(task)
            elif task_type == "consistency_check":
                result = await self._check_consistency(task)
            elif task_type == "format_validation":
                result = await self._validate_format(task)
            else:
                result = await self._general_validation(task)
            
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
    
    async def _validate_content(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content quality and accuracy"""
        content = task.get("content", "")
        validation_criteria = task.get("criteria", ["completeness", "accuracy", "relevance"])
        
        self.update_state("working", "content_validation", 0.3)
        
        validation_results = {}
        overall_issues = []
        overall_suggestions = []
        
        # Validate based on criteria
        for criterion in validation_criteria:
            if criterion == "completeness":
                result = await self._validate_completeness(content)
            elif criterion == "accuracy":
                result = await self._validate_accuracy(content, task)
            elif criterion == "relevance":
                result = await self._validate_relevance(content, task)
            elif criterion == "readability":
                result = await self._validate_readability(content)
            elif criterion == "structure":
                result = await self._validate_structure(content)
            else:
                result = ValidationResult(True, 0.5, [], [f"Unknown validation criterion: {criterion}"])
            
            validation_results[criterion] = {
                "is_valid": result.is_valid,
                "confidence": result.confidence,
                "issues": result.issues,
                "suggestions": result.suggestions
            }
            
            overall_issues.extend(result.issues)
            overall_suggestions.extend(result.suggestions)
        
        self.update_state("working", "content_validation", 0.8)
        
        # Calculate overall score
        overall_score = await self._calculate_overall_score(validation_results)
        
        return {
            "validation_type": "content_validation",
            "overall_score": overall_score,
            "is_valid": overall_score >= self.quality_threshold,
            "criteria_results": validation_results,
            "summary": {
                "total_issues": len(overall_issues),
                "total_suggestions": len(overall_suggestions),
                "validation_criteria": len(validation_criteria)
            },
            "issues": overall_issues[:self.max_issues_reported],
            "suggestions": overall_suggestions[:self.max_issues_reported]
        }
    
    async def _validate_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entire workflow results"""
        workflow_result = task.get("workflow_result", {})
        
        self.update_state("working", "workflow_validation", 0.4)
        
        validation_results = {
            "stage_validation": {},
            "consistency_validation": {},
            "completeness_validation": {}
        }
        
        # Validate each stage
        stage_results = workflow_result.get("stage_results", {})
        for stage_name, stage_result in stage_results.items():
            validation_results["stage_validation"][stage_name] = await self._validate_stage_result(
                stage_name, stage_result
            )
        
        self.update_state("working", "workflow_validation", 0.7)
        
        # Validate consistency across stages
        validation_results["consistency_validation"] = await self._validate_cross_stage_consistency(
            stage_results
        )
        
        # Validate completeness
        validation_results["completeness_validation"] = await self._validate_workflow_completeness(
            workflow_result
        )
        
        # Generate overall assessment
        overall_assessment = await self._generate_workflow_assessment(validation_results)
        
        return {
            "validation_type": "workflow_validation",
            "overall_assessment": overall_assessment,
            "stage_validations": validation_results["stage_validation"],
            "consistency_score": validation_results["consistency_validation"]["score"],
            "completeness_score": validation_results["completeness_validation"]["score"],
            "recommendations": await self._generate_workflow_recommendations(validation_results)
        }
    
    async def _assess_quality(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality of results"""
        data = task.get("data", {})
        quality_dimensions = task.get("dimensions", ["accuracy", "completeness", "clarity", "relevance"])
        
        self.update_state("working", "quality_assessment", 0.4)
        
        quality_scores = {}
        
        for dimension in quality_dimensions:
            score = await self._assess_quality_dimension(data, dimension)
            quality_scores[dimension] = score
        
        # Calculate weighted overall score
        weights = {
            "accuracy": 0.3,
            "completeness": 0.25,
            "clarity": 0.2,
            "relevance": 0.25
        }
        
        overall_score = sum(
            quality_scores.get(dim, 0) * weights.get(dim, 0.2)
            for dim in quality_dimensions
        )
        
        quality_grade = await self._determine_quality_grade(overall_score)
        
        return {
            "assessment_type": "quality_assessment",
            "overall_score": overall_score,
            "quality_grade": quality_grade,
            "dimension_scores": quality_scores,
            "meets_threshold": overall_score >= self.quality_threshold,
            "improvement_areas": [
                dim for dim, score in quality_scores.items()
                if score < self.quality_threshold
            ]
        }
    
    async def _check_consistency(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency across multiple sources or results"""
        sources = task.get("sources", [])
        consistency_type = task.get("consistency_type", "content")
        
        self.update_state("working", "consistency_check", 0.4)
        
        if consistency_type == "content":
            consistency_result = await self._check_content_consistency(sources)
        elif consistency_type == "format":
            consistency_result = await self._check_format_consistency(sources)
        elif consistency_type == "factual":
            consistency_result = await self._check_factual_consistency(sources)
        else:
            consistency_result = await self._check_general_consistency(sources)
        
        return {
            "consistency_type": consistency_type,
            "consistency_score": consistency_result["score"],
            "is_consistent": consistency_result["score"] >= self.consistency_threshold,
            "inconsistencies": consistency_result.get("inconsistencies", []),
            "consistency_details": consistency_result
        }
    
    async def _validate_format(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate format and structure"""
        content = task.get("content", "")
        expected_format = task.get("format", "markdown")
        
        self.update_state("working", "format_validation", 0.5)
        
        format_validation = await self._check_format_compliance(content, expected_format)
        structure_validation = await self._check_structure_quality(content)
        
        return {
            "validation_type": "format_validation",
            "expected_format": expected_format,
            "format_compliance": format_validation,
            "structure_quality": structure_validation,
            "is_valid": format_validation["is_valid"] and structure_validation["is_valid"],
            "validation_details": {
                "format_issues": format_validation.get("issues", []),
                "structure_issues": structure_validation.get("issues", [])
            }
        }
    
    async def _general_validation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General validation fallback method"""
        return await self._validate_content(task)
    
    async def _validate_completeness(self, content: str) -> ValidationResult:
        """Validate content completeness"""
        issues = []
        suggestions = []
        
        # Check minimum length
        if len(content) < 100:
            issues.append("Content is too short (less than 100 characters)")
            suggestions.append("Add more detailed information")
        
        # Check for common sections in reports
        sections = ["summary", "introduction", "conclusion", "result"]
        found_sections = sum(1 for section in sections if section.lower() in content.lower())
        
        if found_sections < 2:
            issues.append("Missing common report sections")
            suggestions.append("Consider adding sections like summary, introduction, or conclusion")
        
        # Check for empty lines (structure indicator)
        if content.count('\n\n') < 2:
            issues.append("Lacks proper paragraph structure")
            suggestions.append("Add proper paragraph breaks for better readability")
        
        confidence = max(0.3, 1.0 - (len(issues) * 0.2))
        is_valid = len(issues) == 0 or confidence > 0.6
        
        return ValidationResult(is_valid, confidence, issues, suggestions)
    
    async def _validate_accuracy(self, content: str, task: Dict[str, Any]) -> ValidationResult:
        """Validate content accuracy"""
        issues = []
        suggestions = []
        
        # Check for unsupported claims
        claim_indicators = ["definitely", "certainly", "always", "never", "all", "none"]
        for indicator in claim_indicators:
            if indicator in content.lower():
                issues.append(f"Contains absolute claim: '{indicator}' - may lack nuance")
        
        # Check for citation patterns
        citation_patterns = [r"\[ref_\d+\]", r"\[\d+\]", r"\([^)]*\d{4}[^)]*\)"]
        has_citations = any(re.search(pattern, content) for pattern in citation_patterns)
        
        if len(content) > 500 and not has_citations:
            issues.append("Long content lacks citations or references")
            suggestions.append("Add proper citations to support claims")
        
        # Check for hedge words (indicates uncertainty, which can be good for accuracy)
        hedge_words = ["may", "might", "could", "possibly", "likely", "suggests"]
        hedge_count = sum(1 for word in hedge_words if word in content.lower())
        
        if hedge_count == 0 and len(content) > 200:
            suggestions.append("Consider using hedge words to indicate uncertainty where appropriate")
        
        confidence = max(0.4, 1.0 - (len(issues) * 0.15))
        is_valid = len(issues) <= 2
        
        return ValidationResult(is_valid, confidence, issues, suggestions)
    
    async def _validate_relevance(self, content: str, task: Dict[str, Any]) -> ValidationResult:
        """Validate content relevance to query/topic"""
        issues = []
        suggestions = []
        
        query = task.get("query", "")
        topic_keywords = task.get("topic_keywords", [])
        
        if query:
            # Check if query terms appear in content
            query_words = query.lower().split()
            content_lower = content.lower()
            
            matching_words = sum(1 for word in query_words if word in content_lower)
            relevance_ratio = matching_words / len(query_words) if query_words else 0
            
            if relevance_ratio < 0.3:
                issues.append(f"Low relevance to query '{query}' (only {relevance_ratio:.1%} query terms found)")
                suggestions.append("Ensure content directly addresses the main query")
        
        if topic_keywords:
            # Check topic keyword coverage
            keyword_coverage = sum(1 for keyword in topic_keywords if keyword.lower() in content.lower())
            coverage_ratio = keyword_coverage / len(topic_keywords) if topic_keywords else 0
            
            if coverage_ratio < 0.5:
                issues.append(f"Limited topic keyword coverage ({coverage_ratio:.1%})")
                suggestions.append("Include more topic-specific terminology")
        
        confidence = max(0.5, 1.0 - (len(issues) * 0.3))
        is_valid = len(issues) == 0
        
        return ValidationResult(is_valid, confidence, issues, suggestions)
    
    async def _validate_readability(self, content: str) -> ValidationResult:
        """Validate content readability"""
        issues = []
        suggestions = []
        
        # Average sentence length
        sentences = content.split('. ')
        if sentences:
            avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
            
            if avg_sentence_length > 25:
                issues.append("Sentences are too long on average")
                suggestions.append("Break down long sentences for better readability")
            elif avg_sentence_length < 8:
                issues.append("Sentences are too short on average")
                suggestions.append("Consider combining short sentences for better flow")
        
        # Paragraph length
        paragraphs = content.split('\n\n')
        if paragraphs:
            avg_paragraph_length = sum(len(paragraph.split()) for paragraph in paragraphs) / len(paragraphs)
            
            if avg_paragraph_length > 150:
                issues.append("Paragraphs are too long")
                suggestions.append("Break down long paragraphs")
        
        # Check for headers/structure
        if len(content) > 500 and not re.search(r'^#{1,6}\s', content, re.MULTILINE):
            issues.append("Long content lacks headers for navigation")
            suggestions.append("Add headers to improve document structure")
        
        confidence = max(0.6, 1.0 - (len(issues) * 0.2))
        is_valid = len(issues) <= 1
        
        return ValidationResult(is_valid, confidence, issues, suggestions)
    
    async def _validate_structure(self, content: str) -> ValidationResult:
        """Validate content structure"""
        issues = []
        suggestions = []
        
        # Check for logical flow
        if len(content) > 300:
            # Look for introduction, body, conclusion pattern
            has_intro = any(word in content[:200].lower() for word in ["introduction", "overview", "summary"])
            has_conclusion = any(word in content[-200:].lower() for word in ["conclusion", "summary", "recommendations"])
            
            if not has_intro:
                suggestions.append("Consider adding an introduction or overview")
            
            if not has_conclusion:
                suggestions.append("Consider adding a conclusion or summary")
        
        # Check for consistent formatting
        header_pattern = r'^#{1,6}\s'
        headers = re.findall(header_pattern, content, re.MULTILINE)
        
        if len(content) > 800 and len(headers) < 2:
            issues.append("Long content lacks sufficient headers")
            suggestions.append("Add more section headers for better organization")
        
        confidence = max(0.7, 1.0 - (len(issues) * 0.2))
        is_valid = len(issues) == 0
        
        return ValidationResult(is_valid, confidence, issues, suggestions)
    
    async def _calculate_overall_score(self, validation_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall validation score"""
        total_confidence = 0
        count = 0
        
        for criterion, result in validation_results.items():
            if "confidence" in result:
                total_confidence += result["confidence"]
                count += 1
        
        return total_confidence / count if count > 0 else 0.0
    
    async def _validate_stage_result(self, stage_name: str, stage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual stage result"""
        issues = []
        
        # Check if stage completed successfully
        if stage_result.get("status") != "success":
            issues.append(f"Stage {stage_name} did not complete successfully")
        
        # Check processing time
        processing_time = stage_result.get("processing_time", 0)
        if processing_time > 300:  # 5 minutes
            issues.append(f"Stage {stage_name} took unusually long ({processing_time:.1f}s)")
        
        # Check for result content
        result_data = stage_result.get("result", {})
        if not result_data:
            issues.append(f"Stage {stage_name} produced no result data")
        
        confidence = max(0.3, 1.0 - (len(issues) * 0.3))
        
        return {
            "is_valid": len(issues) == 0,
            "confidence": confidence,
            "issues": issues,
            "processing_time": processing_time
        }
    
    async def _validate_cross_stage_consistency(self, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency across workflow stages"""
        consistency_issues = []
        
        # Check if query/topic is consistent across stages
        queries = set()
        for stage_result in stage_results.values():
            result_data = stage_result.get("result", {})
            if "query" in result_data:
                queries.add(result_data["query"])
        
        if len(queries) > 1:
            consistency_issues.append("Different queries used across stages")
        
        # Check data flow consistency
        research_count = 0
        analysis_count = 0
        
        for stage_name, stage_result in stage_results.items():
            result_data = stage_result.get("result", {})
            
            if "research" in stage_name:
                research_count = len(result_data.get("results", []))
            elif "analysis" in stage_name:
                if "research_data" in result_data:
                    analysis_count = len(result_data["research_data"].get("results", []))
        
        if research_count > 0 and analysis_count > 0 and research_count != analysis_count:
            consistency_issues.append("Inconsistent data count between research and analysis stages")
        
        score = max(0.5, 1.0 - (len(consistency_issues) * 0.2))
        
        return {
            "score": score,
            "is_consistent": len(consistency_issues) == 0,
            "inconsistencies": consistency_issues
        }
    
    async def _validate_workflow_completeness(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow completeness"""
        issues = []
        
        # Check required components
        required_components = ["stage_results", "final_output", "execution_summary"]
        for component in required_components:
            if component not in workflow_result:
                issues.append(f"Missing required component: {component}")
        
        # Check stage coverage
        stage_results = workflow_result.get("stage_results", {})
        if not stage_results:
            issues.append("No stage results found")
        
        # Check final output
        final_output = workflow_result.get("final_output")
        if not final_output:
            issues.append("No final output generated")
        
        score = max(0.4, 1.0 - (len(issues) * 0.25))
        
        return {
            "score": score,
            "is_complete": len(issues) == 0,
            "missing_components": issues
        }
    
    async def _generate_workflow_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall workflow assessment"""
        stage_validation = validation_results["stage_validation"]
        consistency_score = validation_results["consistency_validation"]["score"]
        completeness_score = validation_results["completeness_validation"]["score"]
        
        # Calculate average stage confidence
        stage_confidences = [v["confidence"] for v in stage_validation.values()]
        avg_stage_confidence = sum(stage_confidences) / len(stage_confidences) if stage_confidences else 0
        
        # Overall score
        overall_score = (avg_stage_confidence + consistency_score + completeness_score) / 3
        
        # Determine grade
        if overall_score >= 0.9:
            grade = "Excellent"
        elif overall_score >= 0.8:
            grade = "Good"
        elif overall_score >= 0.7:
            grade = "Satisfactory"
        elif overall_score >= 0.6:
            grade = "Needs Improvement"
        else:
            grade = "Poor"
        
        return {
            "overall_score": overall_score,
            "grade": grade,
            "meets_standards": overall_score >= self.quality_threshold,
            "component_scores": {
                "stage_quality": avg_stage_confidence,
                "consistency": consistency_score,
                "completeness": completeness_score
            }
        }
    
    async def _generate_workflow_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for workflow improvement"""
        recommendations = []
        
        # Stage-specific recommendations
        stage_validation = validation_results["stage_validation"]
        for stage_name, validation in stage_validation.items():
            if validation["confidence"] < 0.7:
                recommendations.append(f"Improve {stage_name} stage quality")
        
        # Consistency recommendations
        consistency_validation = validation_results["consistency_validation"]
        if consistency_validation["score"] < 0.8:
            recommendations.append("Improve cross-stage consistency")
        
        # Completeness recommendations
        completeness_validation = validation_results["completeness_validation"]
        if completeness_validation["score"] < 0.8:
            recommendations.append("Ensure all workflow components are present")
        
        return recommendations
    
    async def _assess_quality_dimension(self, data: Dict[str, Any], dimension: str) -> float:
        """Assess quality for a specific dimension"""
        if dimension == "accuracy":
            # Simple accuracy assessment based on confidence scores
            confidence_scores = []
            if "confidence_scores" in data:
                confidence_scores = list(data["confidence_scores"].values())
            return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        elif dimension == "completeness":
            # Assess based on data presence
            score = 0.5
            if "results" in data and data["results"]:
                score += 0.3
            if "analysis" in data and data["analysis"]:
                score += 0.2
            return min(score, 1.0)
        
        elif dimension == "clarity":
            # Assess based on structure and formatting
            if "content" in data:
                content = str(data["content"])
                # Simple clarity score based on structure
                has_headers = bool(re.search(r'^#{1,6}\s', content, re.MULTILINE))
                has_paragraphs = '\n\n' in content
                reasonable_length = 100 <= len(content) <= 5000
                
                score = 0.4
                if has_headers:
                    score += 0.2
                if has_paragraphs:
                    score += 0.2
                if reasonable_length:
                    score += 0.2
                
                return score
            return 0.5
        
        elif dimension == "relevance":
            # Assess based on query matching
            if "query" in data:
                # Simple relevance assessment
                return 0.7  # Default relevance score
            return 0.5
        
        else:
            return 0.5  # Default score for unknown dimensions
    
    async def _determine_quality_grade(self, score: float) -> str:
        """Determine quality grade from score"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    async def _check_content_consistency(self, sources: List[Any]) -> Dict[str, Any]:
        """Check consistency across content sources"""
        if len(sources) < 2:
            return {"score": 1.0, "note": "Insufficient sources for comparison"}
        
        # Simple consistency check based on content overlap
        if all(isinstance(source, dict) for source in sources):
            contents = [str(source.get("content", "")) for source in sources]
        else:
            contents = [str(source) for source in sources]
        
        # Calculate overlap (simplified)
        word_sets = [set(content.lower().split()) for content in contents]
        
        if not word_sets or not any(word_sets):
            return {"score": 0.0, "inconsistencies": ["No content to compare"]}
        
        # Calculate Jaccard similarity for pairs
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        inconsistencies = []
        if avg_similarity < 0.3:
            inconsistencies.append("Low content overlap between sources")
        
        return {
            "score": avg_similarity,
            "average_similarity": avg_similarity,
            "inconsistencies": inconsistencies
        }
    
    async def _check_format_consistency(self, sources: List[Any]) -> Dict[str, Any]:
        """Check format consistency across sources"""
        format_types = set()
        
        for source in sources:
            if isinstance(source, dict):
                format_types.add(source.get("format", "unknown"))
            else:
                # Try to detect format
                content = str(source)
                if content.startswith("#"):
                    format_types.add("markdown")
                elif "<" in content and ">" in content:
                    format_types.add("html")
                else:
                    format_types.add("plain_text")
        
        is_consistent = len(format_types) <= 1
        score = 1.0 if is_consistent else 0.5
        
        inconsistencies = []
        if not is_consistent:
            inconsistencies.append(f"Multiple formats detected: {list(format_types)}")
        
        return {
            "score": score,
            "is_consistent": is_consistent,
            "formats_detected": list(format_types),
            "inconsistencies": inconsistencies
        }
    
    async def _check_factual_consistency(self, sources: List[Any]) -> Dict[str, Any]:
        """Check factual consistency across sources"""
        # Simplified factual consistency check
        # In practice, this would involve NLP and fact-checking
        
        return {
            "score": 0.75,  # Default score
            "note": "Factual consistency checking requires advanced NLP capabilities",
            "inconsistencies": []
        }
    
    async def _check_general_consistency(self, sources: List[Any]) -> Dict[str, Any]:
        """General consistency check"""
        return await self._check_content_consistency(sources)
    
    async def _check_format_compliance(self, content: str, expected_format: str) -> Dict[str, Any]:
        """Check if content complies with expected format"""
        issues = []
        
        if expected_format == "markdown":
            # Check for markdown patterns
            has_headers = bool(re.search(r'^#{1,6}\s', content, re.MULTILINE))
            has_bold = bool(re.search(r'\*\*.+\*\*', content))
            has_structure = '\n\n' in content
            
            if not has_headers and len(content) > 300:
                issues.append("Missing markdown headers")
            
            if not has_structure:
                issues.append("Lacks proper paragraph structure")
        
        elif expected_format == "html":
            # Check for HTML patterns
            has_tags = bool(re.search(r'<[^>]+>', content))
            if not has_tags:
                issues.append("No HTML tags found")
        
        is_valid = len(issues) == 0
        confidence = max(0.5, 1.0 - (len(issues) * 0.3))
        
        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "issues": issues,
            "expected_format": expected_format
        }
    
    async def _check_structure_quality(self, content: str) -> Dict[str, Any]:
        """Check structural quality of content"""
        issues = []
        
        # Check for reasonable paragraph length
        paragraphs = content.split('\n\n')
        if paragraphs:
            avg_para_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            if avg_para_length > 200:
                issues.append("Paragraphs are too long")
            elif avg_para_length < 20 and len(paragraphs) > 3:
                issues.append("Paragraphs are too short")
        
        # Check for logical flow indicators
        transition_words = ["however", "therefore", "furthermore", "moreover", "additionally"]
        has_transitions = any(word in content.lower() for word in transition_words)
        
        if len(content) > 500 and not has_transitions:
            issues.append("Lacks transition words for flow")
        
        is_valid = len(issues) == 0
        confidence = max(0.6, 1.0 - (len(issues) * 0.2))
        
        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "issues": issues
        }
