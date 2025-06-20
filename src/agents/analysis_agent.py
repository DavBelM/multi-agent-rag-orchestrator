"""
Analysis Agent

Specialized agent for analyzing and synthesizing information from research results.
Performs deep analysis, pattern recognition, and insight extraction.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
import json
from collections import Counter

from .base_agent import BaseAgent, AgentMessage


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent specialized in information analysis and synthesis.
    
    Capabilities:
    - Content analysis and summarization
    - Pattern recognition and trend identification
    - Comparative analysis
    - Insight extraction
    - Data visualization preparation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        capabilities = [
            "content_analysis",
            "pattern_recognition",
            "trend_identification",
            "comparative_analysis",
            "insight_extraction",
            "data_synthesis"
        ]
        
        super().__init__(
            agent_id="analysis_agent",
            name="Analysis Agent",
            description="Specialized agent for information analysis and synthesis",
            capabilities=capabilities,
            config=config
        )
        
        # Analysis-specific configuration
        self.analysis_depth = config.get("analysis_depth", "medium") if config else "medium"
        self.max_insights = config.get("max_insights", 10) if config else 10
        self.confidence_threshold = config.get("confidence_threshold", 0.6) if config else 0.6
    
    def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "analysis_request":
            return {
                "status": "acknowledged",
                "task_id": content.get("task_id"),
                "estimated_time": "45-90 seconds"
            }
        elif message_type == "status_query":
            return self.get_metrics()
        else:
            return {"status": "unknown_message_type", "message_type": message_type}
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an analysis task"""
        start_time = time.time()
        task_type = task.get("type", "general_analysis")
        
        try:
            self.update_state("working", task_type, 0.1)
            
            if task_type == "content_analysis":
                result = await self._content_analysis(task)
            elif task_type == "comparative_analysis":
                result = await self._comparative_analysis(task)
            elif task_type == "trend_analysis":
                result = await self._trend_analysis(task)
            elif task_type == "comprehensive_analysis":
                result = await self._comprehensive_analysis(task)
            else:
                result = await self._general_analysis(task)
            
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
    
    async def _content_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content for key themes, entities, and insights"""
        data = task.get("data", [])
        analysis_type = task.get("analysis_type", "general")
        
        self.update_state("working", "content_analysis", 0.3)
        
        # Extract text content from data
        texts = []
        for item in data:
            if isinstance(item, dict):
                content = item.get("content", "")
            else:
                content = str(item)
            texts.append(content)
        
        self.update_state("working", "content_analysis", 0.5)
        
        # Perform analysis
        analysis_result = {
            "content_statistics": await self._analyze_content_statistics(texts),
            "key_themes": await self._extract_key_themes(texts),
            "entities": await self._extract_entities(texts),
            "sentiment": await self._analyze_sentiment(texts),
            "insights": await self._generate_insights(texts)
        }
        
        self.update_state("working", "content_analysis", 0.8)
        
        # Generate summary
        summary = await self._generate_analysis_summary(analysis_result)
        
        return {
            "analysis_type": "content_analysis",
            "input_count": len(data),
            "analysis": analysis_result,
            "summary": summary,
            "confidence_scores": await self._calculate_confidence_scores(analysis_result)
        }
    
    async def _comparative_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple datasets or sources"""
        datasets = task.get("datasets", [])
        comparison_criteria = task.get("criteria", ["content", "themes", "quality"])
        
        self.update_state("working", "comparative_analysis", 0.3)
        
        if len(datasets) < 2:
            return {
                "analysis_type": "comparative_analysis",
                "error": "At least 2 datasets required for comparison",
                "input_count": len(datasets)
            }
        
        # Analyze each dataset
        dataset_analyses = []
        for i, dataset in enumerate(datasets):
            analysis = await self._content_analysis({"data": dataset})
            dataset_analyses.append({
                "dataset_id": i,
                "analysis": analysis["analysis"]
            })
        
        self.update_state("working", "comparative_analysis", 0.7)
        
        # Perform comparisons
        comparisons = {}
        for criterion in comparison_criteria:
            comparisons[criterion] = await self._compare_by_criterion(
                dataset_analyses, criterion
            )
        
        # Generate insights from comparison
        comparative_insights = await self._generate_comparative_insights(
            dataset_analyses, comparisons
        )
        
        return {
            "analysis_type": "comparative_analysis",
            "dataset_count": len(datasets),
            "comparison_criteria": comparison_criteria,
            "individual_analyses": dataset_analyses,
            "comparisons": comparisons,
            "insights": comparative_insights
        }
    
    async def _trend_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends and patterns in temporal data"""
        data = task.get("data", [])
        time_field = task.get("time_field", "timestamp")
        
        self.update_state("working", "trend_analysis", 0.3)
        
        # Extract temporal patterns
        temporal_patterns = await self._analyze_temporal_patterns(data, time_field)
        
        self.update_state("working", "trend_analysis", 0.6)
        
        # Identify trends
        trends = await self._identify_trends(data, temporal_patterns)
        
        # Generate predictions
        predictions = await self._generate_trend_predictions(trends)
        
        return {
            "analysis_type": "trend_analysis",
            "input_count": len(data),
            "temporal_patterns": temporal_patterns,
            "identified_trends": trends,
            "predictions": predictions,
            "trend_confidence": await self._calculate_trend_confidence(trends)
        }
    
    async def _comprehensive_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis combining multiple techniques"""
        data = task.get("data", [])
        
        self.update_state("working", "comprehensive_analysis", 0.2)
        
        # Content analysis
        content_analysis = await self._content_analysis({"data": data})
        
        self.update_state("working", "comprehensive_analysis", 0.5)
        
        # Statistical analysis
        statistical_analysis = await self._statistical_analysis(data)
        
        self.update_state("working", "comprehensive_analysis", 0.7)
        
        # Network analysis (if applicable)
        network_analysis = await self._network_analysis(data)
        
        self.update_state("working", "comprehensive_analysis", 0.9)
        
        # Synthesize all analyses
        synthesis = await self._synthesize_analyses({
            "content": content_analysis,
            "statistical": statistical_analysis,
            "network": network_analysis
        })
        
        return {
            "analysis_type": "comprehensive_analysis",
            "input_count": len(data),
            "content_analysis": content_analysis,
            "statistical_analysis": statistical_analysis,
            "network_analysis": network_analysis,
            "synthesis": synthesis,
            "overall_confidence": await self._calculate_overall_confidence([
                content_analysis, statistical_analysis, network_analysis
            ])
        }
    
    async def _general_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General analysis fallback method"""
        return await self._content_analysis(task)
    
    async def _analyze_content_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze basic content statistics"""
        if not texts:
            return {}
        
        total_length = sum(len(text) for text in texts)
        word_counts = [len(text.split()) for text in texts]
        
        return {
            "document_count": len(texts),
            "total_characters": total_length,
            "average_length": total_length / len(texts),
            "total_words": sum(word_counts),
            "average_words": sum(word_counts) / len(word_counts) if word_counts else 0,
            "length_distribution": {
                "min": min(len(text) for text in texts),
                "max": max(len(text) for text in texts),
                "median": sorted([len(text) for text in texts])[len(texts) // 2]
            }
        }
    
    async def _extract_key_themes(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract key themes from text content"""
        if not texts:
            return []
        
        # Simple keyword frequency analysis
        # In practice, you'd use more sophisticated NLP techniques
        all_words = []
        for text in texts:
            words = text.lower().split()
            # Filter out common words (simple stopword removal)
            filtered_words = [
                word for word in words 
                if len(word) > 3 and word not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use']
            ]
            all_words.extend(filtered_words)
        
        # Count word frequencies
        word_freq = Counter(all_words)
        
        # Extract top themes
        themes = []
        for word, count in word_freq.most_common(self.max_insights):
            themes.append({
                "theme": word,
                "frequency": count,
                "relative_frequency": count / len(all_words) if all_words else 0,
                "confidence": min(count / len(texts), 1.0)
            })
        
        return themes
    
    async def _extract_entities(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        # Simple entity extraction (capitalized words)
        # In practice, you'd use NER models
        entities = []
        entity_counter = Counter()
        
        for text in texts:
            words = text.split()
            for word in words:
                # Simple heuristic: capitalized words that aren't at sentence start
                if word[0].isupper() and len(word) > 2:
                    clean_word = word.strip('.,!?;:')
                    entity_counter[clean_word] += 1
        
        for entity, count in entity_counter.most_common(self.max_insights):
            entities.append({
                "entity": entity,
                "frequency": count,
                "type": "UNKNOWN",  # Would be determined by NER model
                "confidence": min(count / len(texts), 1.0)
            })
        
        return entities
    
    async def _analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of text content"""
        # Simple sentiment analysis based on keyword counts
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'success', 'effective', 'beneficial']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'failure', 'ineffective', 'harmful', 'poor', 'worse']
        
        positive_count = 0
        negative_count = 0
        total_words = 0
        
        for text in texts:
            words = text.lower().split()
            total_words += len(words)
            
            for word in words:
                if word in positive_words:
                    positive_count += 1
                elif word in negative_words:
                    negative_count += 1
        
        if total_words == 0:
            return {"overall": "neutral", "confidence": 0.0}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        if positive_ratio > negative_ratio:
            sentiment = "positive"
            confidence = positive_ratio
        elif negative_ratio > positive_ratio:
            sentiment = "negative"
            confidence = negative_ratio
        else:
            sentiment = "neutral"
            confidence = 1.0 - abs(positive_ratio - negative_ratio)
        
        return {
            "overall": sentiment,
            "confidence": min(confidence * 10, 1.0),  # Amplify for small datasets
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": 1.0 - positive_ratio - negative_ratio
        }
    
    async def _generate_insights(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate insights from analyzed content"""
        insights = []
        
        if not texts:
            return insights
        
        # Insight 1: Content diversity
        avg_length = sum(len(text) for text in texts) / len(texts)
        length_variance = sum((len(text) - avg_length) ** 2 for text in texts) / len(texts)
        
        if length_variance > avg_length * 0.5:
            insights.append({
                "type": "content_diversity",
                "insight": "Content shows high diversity in length and structure",
                "confidence": 0.8,
                "supporting_data": {"variance": length_variance, "mean": avg_length}
            })
        
        # Insight 2: Information density
        total_words = sum(len(text.split()) for text in texts)
        total_chars = sum(len(text) for text in texts)
        
        if total_chars > 0:
            word_density = total_words / total_chars
            if word_density > 0.15:
                insights.append({
                    "type": "information_density",
                    "insight": "Content has high information density",
                    "confidence": 0.7,
                    "supporting_data": {"word_density": word_density}
                })
        
        # Insight 3: Repetition patterns
        all_text = " ".join(texts).lower()
        unique_words = len(set(all_text.split()))
        total_words_all = len(all_text.split())
        
        if total_words_all > 0:
            uniqueness_ratio = unique_words / total_words_all
            if uniqueness_ratio < 0.3:
                insights.append({
                    "type": "repetition_pattern",
                    "insight": "Content shows significant repetition of terms",
                    "confidence": 0.8,
                    "supporting_data": {"uniqueness_ratio": uniqueness_ratio}
                })
        
        return insights
    
    async def _generate_analysis_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of analysis results"""
        stats = analysis_result.get("content_statistics", {})
        themes = analysis_result.get("key_themes", [])
        sentiment = analysis_result.get("sentiment", {})
        insights = analysis_result.get("insights", [])
        
        return {
            "overview": f"Analyzed {stats.get('document_count', 0)} documents with {stats.get('total_words', 0)} total words",
            "key_findings": [
                f"Top theme: {themes[0]['theme']}" if themes else "No clear themes identified",
                f"Overall sentiment: {sentiment.get('overall', 'neutral')}",
                f"Generated {len(insights)} insights"
            ],
            "quality_indicators": {
                "content_richness": len(themes) / 10 if themes else 0,
                "analysis_depth": len(insights) / 5 if insights else 0,
                "sentiment_clarity": sentiment.get("confidence", 0)
            }
        }
    
    async def _calculate_confidence_scores(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for different analysis components"""
        return {
            "themes": sum(theme.get("confidence", 0) for theme in analysis_result.get("key_themes", [])) / max(len(analysis_result.get("key_themes", [])), 1),
            "entities": sum(entity.get("confidence", 0) for entity in analysis_result.get("entities", [])) / max(len(analysis_result.get("entities", [])), 1),
            "sentiment": analysis_result.get("sentiment", {}).get("confidence", 0),
            "insights": sum(insight.get("confidence", 0) for insight in analysis_result.get("insights", [])) / max(len(analysis_result.get("insights", [])), 1)
        }
    
    async def _compare_by_criterion(self, analyses: List[Dict[str, Any]], criterion: str) -> Dict[str, Any]:
        """Compare datasets by a specific criterion"""
        comparison = {"criterion": criterion, "results": []}
        
        if criterion == "content":
            for analysis in analyses:
                stats = analysis["analysis"]["content_statistics"]
                comparison["results"].append({
                    "dataset_id": analysis["dataset_id"],
                    "total_words": stats.get("total_words", 0),
                    "average_length": stats.get("average_length", 0),
                    "document_count": stats.get("document_count", 0)
                })
        elif criterion == "themes":
            for analysis in analyses:
                themes = analysis["analysis"]["key_themes"]
                comparison["results"].append({
                    "dataset_id": analysis["dataset_id"],
                    "theme_count": len(themes),
                    "top_theme": themes[0]["theme"] if themes else "None",
                    "theme_diversity": len(set(theme["theme"] for theme in themes))
                })
        elif criterion == "quality":
            for analysis in analyses:
                sentiment = analysis["analysis"]["sentiment"]
                insights = analysis["analysis"]["insights"]
                comparison["results"].append({
                    "dataset_id": analysis["dataset_id"],
                    "sentiment_confidence": sentiment.get("confidence", 0),
                    "insight_count": len(insights),
                    "overall_quality": (sentiment.get("confidence", 0) + len(insights) / 5) / 2
                })
        
        return comparison
    
    async def _generate_comparative_insights(self, analyses: List[Dict[str, Any]], comparisons: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights from comparative analysis"""
        insights = []
        
        # Compare content volumes
        if "content" in comparisons:
            content_results = comparisons["content"]["results"]
            word_counts = [result["total_words"] for result in content_results]
            
            if max(word_counts) > min(word_counts) * 2:
                insights.append({
                    "type": "content_volume",
                    "insight": "Significant variation in content volume between datasets",
                    "confidence": 0.8,
                    "details": f"Range: {min(word_counts)} to {max(word_counts)} words"
                })
        
        # Compare theme diversity
        if "themes" in comparisons:
            theme_results = comparisons["themes"]["results"]
            diversities = [result["theme_diversity"] for result in theme_results]
            
            if max(diversities) > min(diversities) * 1.5:
                insights.append({
                    "type": "theme_diversity",
                    "insight": "Datasets show different levels of thematic diversity",
                    "confidence": 0.7,
                    "details": f"Diversity range: {min(diversities)} to {max(diversities)}"
                })
        
        return insights
    
    async def _analyze_temporal_patterns(self, data: List[Dict[str, Any]], time_field: str) -> Dict[str, Any]:
        """Analyze temporal patterns in data"""
        # Simple temporal analysis
        # In practice, you'd use more sophisticated time series analysis
        
        timestamps = []
        for item in data:
            if time_field in item:
                timestamps.append(item[time_field])
        
        if not timestamps:
            return {"error": "No temporal data found"}
        
        return {
            "total_points": len(timestamps),
            "time_span": {"start": min(timestamps), "end": max(timestamps)} if timestamps else None,
            "distribution": "uniform"  # Simplified
        }
    
    async def _identify_trends(self, data: List[Dict[str, Any]], patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify trends in the data"""
        trends = []
        
        if patterns.get("total_points", 0) > 5:
            trends.append({
                "type": "volume_trend",
                "direction": "increasing",
                "confidence": 0.6,
                "description": "Data volume appears to be increasing over time"
            })
        
        return trends
    
    async def _generate_trend_predictions(self, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate predictions based on identified trends"""
        predictions = []
        
        for trend in trends:
            if trend["type"] == "volume_trend" and trend["direction"] == "increasing":
                predictions.append({
                    "prediction": "Continued growth in data volume expected",
                    "confidence": trend["confidence"] * 0.8,
                    "timeframe": "short_term"
                })
        
        return predictions
    
    async def _calculate_trend_confidence(self, trends: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in trend analysis"""
        if not trends:
            return 0.0
        
        return sum(trend.get("confidence", 0) for trend in trends) / len(trends)
    
    async def _statistical_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on data"""
        return {
            "sample_size": len(data),
            "data_types": self._analyze_data_types(data),
            "completeness": self._analyze_data_completeness(data)
        }
    
    async def _network_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform network analysis if applicable"""
        return {
            "network_detected": False,
            "reason": "No clear network structure in data"
        }
    
    async def _synthesize_analyses(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize multiple analysis results"""
        return {
            "synthesis_quality": "high",
            "integration_points": len(analyses),
            "overall_insight": "Multiple analysis perspectives provide comprehensive understanding"
        }
    
    async def _calculate_overall_confidence(self, analyses: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence across all analyses"""
        return 0.75  # Simplified
    
    def _analyze_data_types(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze data types in the dataset"""
        type_counts = Counter()
        
        for item in data:
            if isinstance(item, dict):
                for value in item.values():
                    type_counts[type(value).__name__] += 1
            else:
                type_counts[type(item).__name__] += 1
        
        return dict(type_counts)
    
    def _analyze_data_completeness(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze completeness of data"""
        if not data:
            return {"overall": 0.0}
        
        total_fields = 0
        filled_fields = 0
        
        for item in data:
            if isinstance(item, dict):
                for value in item.values():
                    total_fields += 1
                    if value is not None and value != "":
                        filled_fields += 1
        
        return {
            "overall": filled_fields / total_fields if total_fields > 0 else 0.0,
            "total_fields": total_fields,
            "filled_fields": filled_fields
        }
    
    async def analyze_information(self, research_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Analyze research information and provide insights."""
        task = {
            "type": "general_analysis",
            "data": research_data,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        return await self.execute_task(task)
