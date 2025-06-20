"""
Coordinator Agent

Central coordination agent that manages workflow, task distribution, and agent communication.
Acts as the orchestrator's primary interface for complex multi-agent operations.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import uuid
from enum import Enum

from .base_agent import BaseAgent, AgentMessage


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowTask:
    """Represents a task in the workflow"""
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        task_data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[List[str]] = None,
        timeout: Optional[int] = None
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.task_data = task_data
        self.priority = priority
        self.dependencies = dependencies or []
        self.timeout = timeout or 300  # 5 minutes default
        
        self.status = TaskStatus.PENDING
        self.assigned_agent = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent that manages multi-agent workflows and task distribution.
    
    Capabilities:
    - Workflow orchestration and planning
    - Task scheduling and assignment
    - Agent communication and coordination
    - Resource optimization
    - Error handling and recovery
    """
    
    def __init__(
        self,
        available_agents: Dict[str, BaseAgent],
        config: Optional[Dict[str, Any]] = None
    ):
        capabilities = [
            "workflow_orchestration",
            "task_scheduling",
            "agent_coordination",
            "resource_optimization",
            "error_recovery",
            "progress_monitoring"
        ]
        
        super().__init__(
            agent_id="coordinator_agent",
            name="Coordinator Agent",
            description="Central coordination agent for multi-agent workflows",
            capabilities=capabilities,
            config=config
        )
        
        self.available_agents = available_agents
        self.active_tasks: Dict[str, WorkflowTask] = {}
        self.completed_tasks: Dict[str, WorkflowTask] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
        # Coordination-specific configuration
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 5) if config else 5
        self.task_timeout = config.get("task_timeout", 300) if config else 300
        self.retry_attempts = config.get("retry_attempts", 3) if config else 3
    
    def process_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Process incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "workflow_request":
            return {
                "status": "acknowledged",
                "workflow_id": content.get("workflow_id"),
                "estimated_time": "Variable based on complexity"
            }
        elif message_type == "task_status_update":
            return self._handle_task_status_update(content)
        elif message_type == "agent_status_query":
            return self._get_agent_status(content.get("agent_id"))
        elif message_type == "workflow_status_query":
            return self._get_workflow_status(content.get("workflow_id"))
        else:
            return {"status": "unknown_message_type", "message_type": message_type}
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a coordination task (workflow orchestration)"""
        start_time = time.time()
        task_type = task.get("type", "orchestrate_workflow")
        
        try:
            self.update_state("working", task_type, 0.1)
            
            if task_type == "orchestrate_workflow":
                result = await self._orchestrate_workflow(task)
            elif task_type == "schedule_tasks":
                result = await self._schedule_tasks(task)
            elif task_type == "monitor_progress":
                result = await self._monitor_progress(task)
            elif task_type == "handle_failure":
                result = await self._handle_failure(task)
            else:
                result = await self._general_coordination(task)
            
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
    
    async def _orchestrate_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate a complete multi-agent workflow"""
        workflow_config = task.get("workflow_config", {})
        query = task.get("query", "")
        workflow_type = task.get("workflow_type", "comprehensive")
        
        workflow_id = str(uuid.uuid4())
        
        self.update_state("working", "orchestrate_workflow", 0.2)
        
        # Plan the workflow
        workflow_plan = await self._plan_workflow(query, workflow_type, workflow_config)
        
        self.update_state("working", "orchestrate_workflow", 0.3)
        
        # Create and schedule tasks
        tasks = await self._create_workflow_tasks(workflow_plan, workflow_id)
        
        self.update_state("working", "orchestrate_workflow", 0.4)
        
        # Execute the workflow
        workflow_result = await self._execute_workflow(tasks, workflow_id)
        
        self.update_state("working", "orchestrate_workflow", 0.9)
        
        # Record workflow history
        self.workflow_history.append({
            "workflow_id": workflow_id,
            "query": query,
            "workflow_type": workflow_type,
            "tasks_count": len(tasks),
            "start_time": datetime.now().isoformat(),
            "result": workflow_result
        })
        
        return {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "query": query,
            "tasks_executed": len(tasks),
            "result": workflow_result,
            "execution_summary": await self._generate_execution_summary(tasks)
        }
    
    async def _plan_workflow(self, query: str, workflow_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Plan the workflow based on query and type"""
        plan = {
            "workflow_type": workflow_type,
            "query": query,
            "stages": [],
            "estimated_duration": 0
        }
        
        if workflow_type == "comprehensive":
            # Comprehensive workflow: Research -> Analysis -> Writing
            plan["stages"] = [
                {
                    "stage": "research",
                    "agent": "research_agent",
                    "task_type": "comprehensive_research",
                    "estimated_time": 60,
                    "dependencies": []
                },
                {
                    "stage": "analysis",
                    "agent": "analysis_agent",
                    "task_type": "comprehensive_analysis",
                    "estimated_time": 90,
                    "dependencies": ["research"]
                },
                {
                    "stage": "writing",
                    "agent": "writing_agent",
                    "task_type": "synthesis_report",
                    "estimated_time": 120,
                    "dependencies": ["research", "analysis"]
                }
            ]
            plan["estimated_duration"] = 270  # 4.5 minutes
        
        elif workflow_type == "research_focused":
            # Research-focused workflow
            plan["stages"] = [
                {
                    "stage": "research",
                    "agent": "research_agent",
                    "task_type": "comprehensive_research",
                    "estimated_time": 90,
                    "dependencies": []
                },
                {
                    "stage": "summary",
                    "agent": "writing_agent",
                    "task_type": "summary",
                    "estimated_time": 60,
                    "dependencies": ["research"]
                }
            ]
            plan["estimated_duration"] = 150  # 2.5 minutes
        
        elif workflow_type == "analysis_focused":
            # Analysis-focused workflow
            plan["stages"] = [
                {
                    "stage": "research",
                    "agent": "research_agent",
                    "task_type": "semantic_search",
                    "estimated_time": 45,
                    "dependencies": []
                },
                {
                    "stage": "analysis",
                    "agent": "analysis_agent",
                    "task_type": "comprehensive_analysis",
                    "estimated_time": 120,
                    "dependencies": ["research"]
                },
                {
                    "stage": "insights",
                    "agent": "writing_agent",
                    "task_type": "executive_summary",
                    "estimated_time": 45,
                    "dependencies": ["analysis"]
                }
            ]
            plan["estimated_duration"] = 210  # 3.5 minutes
        
        else:
            # Quick workflow
            plan["stages"] = [
                {
                    "stage": "research",
                    "agent": "research_agent",
                    "task_type": "semantic_search",
                    "estimated_time": 30,
                    "dependencies": []
                },
                {
                    "stage": "summary",
                    "agent": "writing_agent",
                    "task_type": "summary",
                    "estimated_time": 30,
                    "dependencies": ["research"]
                }
            ]
            plan["estimated_duration"] = 60  # 1 minute
        
        return plan
    
    async def _create_workflow_tasks(self, workflow_plan: Dict[str, Any], workflow_id: str) -> List[WorkflowTask]:
        """Create workflow tasks from the plan"""
        tasks = []
        query = workflow_plan["query"]
        
        for stage in workflow_plan["stages"]:
            task_id = f"{workflow_id}_{stage['stage']}"
            
            # Prepare task data based on stage
            task_data = {
                "query": query,
                "workflow_id": workflow_id,
                "stage": stage["stage"]
            }
            
            # Add stage-specific data
            if stage["stage"] == "research":
                task_data.update({
                    "type": stage["task_type"],
                    "query": query,
                    "depth": "deep" if "comprehensive" in stage["task_type"] else "medium"
                })
            elif stage["stage"] == "analysis":
                task_data.update({
                    "type": stage["task_type"],
                    "analysis_depth": "deep"
                })
            elif stage["stage"] == "writing":
                task_data.update({
                    "type": stage["task_type"],
                    "style": "professional",
                    "format": "markdown"
                })
            
            # Determine dependencies
            dependencies = []
            for dep in stage.get("dependencies", []):
                dependencies.append(f"{workflow_id}_{dep}")
            
            # Create task
            task = WorkflowTask(
                task_id=task_id,
                task_type=stage["task_type"],
                task_data=task_data,
                priority=TaskPriority.MEDIUM,
                dependencies=dependencies,
                timeout=stage.get("estimated_time", 300)
            )
            
            tasks.append(task)
            self.active_tasks[task_id] = task
        
        return tasks
    
    async def _execute_workflow(self, tasks: List[WorkflowTask], workflow_id: str) -> Dict[str, Any]:
        """Execute the workflow tasks"""
        execution_results = {}
        completed_tasks = set()
        
        # Execute tasks in dependency order
        while len(completed_tasks) < len(tasks):
            # Find tasks ready to execute
            ready_tasks = []
            for task in tasks:
                if (task.task_id not in completed_tasks and
                    task.status == TaskStatus.PENDING and
                    all(dep in completed_tasks for dep in task.dependencies)):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Check for deadlock or all tasks completed
                if len(completed_tasks) == len(tasks):
                    break
                else:
                    self.logger.warning("Potential deadlock detected in workflow")
                    break
            
            # Execute ready tasks (with concurrency limit)
            batch_size = min(len(ready_tasks), self.max_concurrent_tasks)
            batch_tasks = ready_tasks[:batch_size]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(
                *[self._execute_single_task(task, execution_results) for task in batch_tasks],
                return_exceptions=True
            )
            
            # Process results
            for task, result in zip(batch_tasks, batch_results):
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.error = str(result)
                    self.logger.error(f"Task {task.task_id} failed: {result}")
                else:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.result = result
                    execution_results[task.task_id] = result
                
                completed_tasks.add(task.task_id)
                
                # Move to completed tasks
                if task.task_id in self.active_tasks:
                    self.completed_tasks[task.task_id] = self.active_tasks.pop(task.task_id)
        
        # Compile final results
        final_result = await self._compile_workflow_results(execution_results, workflow_id)
        
        return final_result
    
    async def _execute_single_task(self, task: WorkflowTask, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        # Determine which agent to use
        agent = self._get_agent_for_task(task)
        if not agent:
            raise Exception(f"No suitable agent found for task type: {task.task_type}")
        
        # Prepare task data with previous results
        enhanced_task_data = task.task_data.copy()
        
        # Add data from previous stages
        if task.dependencies:
            for dep_task_id in task.dependencies:
                if dep_task_id in previous_results:
                    stage_name = dep_task_id.split("_")[-1]  # Extract stage name
                    enhanced_task_data[f"{stage_name}_data"] = previous_results[dep_task_id]["result"]
        
        # Execute task
        task.assigned_agent = agent.agent_id
        result = await agent.execute_task(enhanced_task_data)
        
        return result
    
    def _get_agent_for_task(self, task: WorkflowTask) -> Optional[BaseAgent]:
        """Get the appropriate agent for a task"""
        task_type = task.task_type
        
        # Map task types to agents
        if any(keyword in task_type for keyword in ["research", "search", "document"]):
            return self.available_agents.get("research_agent")
        elif any(keyword in task_type for keyword in ["analysis", "analyze", "insight"]):
            return self.available_agents.get("analysis_agent")
        elif any(keyword in task_type for keyword in ["writing", "report", "summary"]):
            return self.available_agents.get("writing_agent")
        elif any(keyword in task_type for keyword in ["validation", "verify", "check"]):
            return self.available_agents.get("validation_agent")
        
        return None
    
    async def _compile_workflow_results(self, execution_results: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        """Compile final workflow results"""
        compiled_result = {
            "workflow_id": workflow_id,
            "execution_summary": {
                "total_tasks": len(execution_results),
                "successful_tasks": len([r for r in execution_results.values() if r.get("status") == "success"]),
                "failed_tasks": len([r for r in execution_results.values() if r.get("status") == "error"])
            },
            "stage_results": {},
            "final_output": None
        }
        
        # Organize results by stage
        for task_id, result in execution_results.items():
            stage = task_id.split("_")[-1]
            compiled_result["stage_results"][stage] = result
        
        # Determine final output (usually from writing stage or last successful stage)
        if "writing" in compiled_result["stage_results"]:
            compiled_result["final_output"] = compiled_result["stage_results"]["writing"]
        elif "summary" in compiled_result["stage_results"]:
            compiled_result["final_output"] = compiled_result["stage_results"]["summary"]
        elif "insights" in compiled_result["stage_results"]:
            compiled_result["final_output"] = compiled_result["stage_results"]["insights"]
        else:
            # Use the last successful result
            successful_results = [r for r in execution_results.values() if r.get("status") == "success"]
            if successful_results:
                compiled_result["final_output"] = successful_results[-1]
        
        return compiled_result
    
    async def _schedule_tasks(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule tasks for execution"""
        tasks_to_schedule = task.get("tasks", [])
        
        scheduled_tasks = []
        for task_data in tasks_to_schedule:
            task_id = str(uuid.uuid4())
            workflow_task = WorkflowTask(
                task_id=task_id,
                task_type=task_data.get("type", "general"),
                task_data=task_data,
                priority=TaskPriority(task_data.get("priority", 2))
            )
            
            self.active_tasks[task_id] = workflow_task
            scheduled_tasks.append(task_id)
        
        return {
            "scheduled_tasks": scheduled_tasks,
            "total_scheduled": len(scheduled_tasks)
        }
    
    async def _monitor_progress(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor progress of active tasks"""
        workflow_id = task.get("workflow_id")
        
        active_count = len(self.active_tasks)
        completed_count = len(self.completed_tasks)
        
        # Get status of workflow tasks
        workflow_tasks = []
        if workflow_id:
            workflow_tasks = [
                task for task in {**self.active_tasks, **self.completed_tasks}.values()
                if task.task_data.get("workflow_id") == workflow_id
            ]
        
        return {
            "active_tasks": active_count,
            "completed_tasks": completed_count,
            "workflow_tasks": len(workflow_tasks),
            "agent_status": {
                agent_id: agent.get_state().status
                for agent_id, agent in self.available_agents.items()
            }
        }
    
    async def _handle_failure(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task or workflow failures"""
        failed_task_id = task.get("task_id")
        failure_type = task.get("failure_type", "task_failure")
        
        if failed_task_id and failed_task_id in self.active_tasks:
            failed_task = self.active_tasks[failed_task_id]
            
            # Attempt recovery based on failure type
            if failure_type == "timeout":
                # Retry with extended timeout
                failed_task.timeout += 120  # Add 2 minutes
                failed_task.status = TaskStatus.PENDING
                return {"recovery_action": "timeout_extended", "task_id": failed_task_id}
            
            elif failure_type == "agent_error":
                # Try different agent if available
                alternative_agent = self._find_alternative_agent(failed_task)
                if alternative_agent:
                    failed_task.assigned_agent = alternative_agent.agent_id
                    failed_task.status = TaskStatus.PENDING
                    return {"recovery_action": "agent_reassigned", "task_id": failed_task_id}
            
            # Mark as failed if no recovery possible
            failed_task.status = TaskStatus.FAILED
            self.completed_tasks[failed_task_id] = self.active_tasks.pop(failed_task_id)
            
            return {"recovery_action": "task_failed", "task_id": failed_task_id}
        
        return {"recovery_action": "no_action", "reason": "task_not_found"}
    
    async def _general_coordination(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General coordination fallback method"""
        return await self._monitor_progress(task)
    
    def _find_alternative_agent(self, task: WorkflowTask) -> Optional[BaseAgent]:
        """Find an alternative agent for a failed task"""
        # Simple fallback logic - in practice, this would be more sophisticated
        current_agent = task.assigned_agent
        
        for agent_id, agent in self.available_agents.items():
            if agent_id != current_agent and agent.get_state().status == "idle":
                # Check if agent has relevant capabilities
                required_capabilities = self._get_required_capabilities(task.task_type)
                agent_capabilities = agent.get_capabilities()
                
                if any(cap in agent_capabilities for cap in required_capabilities):
                    return agent
        
        return None
    
    def _get_required_capabilities(self, task_type: str) -> List[str]:
        """Get required capabilities for a task type"""
        if "research" in task_type:
            return ["document_search", "semantic_retrieval"]
        elif "analysis" in task_type:
            return ["content_analysis", "pattern_recognition"]
        elif "writing" in task_type or "summary" in task_type:
            return ["content_generation", "text_synthesis"]
        else:
            return []
    
    def _handle_task_status_update(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task status updates from agents"""
        task_id = content.get("task_id")
        new_status = content.get("status")
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus(new_status)
            
            return {"status": "updated", "task_id": task_id}
        
        return {"status": "task_not_found", "task_id": task_id}
    
    def _get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of a specific agent"""
        if agent_id in self.available_agents:
            agent = self.available_agents[agent_id]
            return {
                "agent_id": agent_id,
                "status": agent.get_state().status,
                "metrics": agent.get_metrics()
            }
        
        return {"agent_id": agent_id, "status": "not_found"}
    
    def _get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow"""
        workflow_tasks = [
            task for task in {**self.active_tasks, **self.completed_tasks}.values()
            if task.task_data.get("workflow_id") == workflow_id
        ]
        
        if not workflow_tasks:
            return {"workflow_id": workflow_id, "status": "not_found"}
        
        total_tasks = len(workflow_tasks)
        completed_tasks = len([t for t in workflow_tasks if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in workflow_tasks if t.status == TaskStatus.FAILED])
        
        if completed_tasks == total_tasks:
            status = "completed"
        elif failed_tasks > 0:
            status = "partially_failed"
        elif any(t.status == TaskStatus.IN_PROGRESS for t in workflow_tasks):
            status = "in_progress"
        else:
            status = "pending"
        
        return {
            "workflow_id": workflow_id,
            "status": status,
            "progress": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks
        }
    
    async def _generate_execution_summary(self, tasks: List[WorkflowTask]) -> Dict[str, Any]:
        """Generate execution summary for completed workflow"""
        total_time = 0
        successful_tasks = 0
        
        for task in tasks:
            if task.started_at and task.completed_at:
                task_time = (task.completed_at - task.started_at).total_seconds()
                total_time += task_time
            
            if task.status == TaskStatus.COMPLETED:
                successful_tasks += 1
        
        return {
            "total_execution_time": total_time,
            "average_task_time": total_time / len(tasks) if tasks else 0,
            "success_rate": successful_tasks / len(tasks) if tasks else 0,
            "task_breakdown": {
                task.task_id: {
                    "status": task.status.value,
                    "execution_time": (task.completed_at - task.started_at).total_seconds() if task.started_at and task.completed_at else 0,
                    "agent": task.assigned_agent
                }
                for task in tasks
            }
        }
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        return self.workflow_history
    
    def get_active_workflows(self) -> Dict[str, Any]:
        """Get currently active workflows"""
        active_workflows = {}
        
        for task in self.active_tasks.values():
            workflow_id = task.task_data.get("workflow_id")
            if workflow_id:
                if workflow_id not in active_workflows:
                    active_workflows[workflow_id] = {
                        "workflow_id": workflow_id,
                        "tasks": [],
                        "status": "active"
                    }
                
                active_workflows[workflow_id]["tasks"].append({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": task.status.value,
                    "assigned_agent": task.assigned_agent
                })
        
        return active_workflows
