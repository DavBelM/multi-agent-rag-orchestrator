"""
FastAPI Web Interface for Multi-Agent RAG Orchestrator

REST API interface for the multi-agent RAG system.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import uuid
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.orchestrator import MultiAgentRAGOrchestrator
from src.agents import ResearchAgent, AnalysisAgent, WritingAgent, CoordinatorAgent, ValidationAgent
from src.core import VectorStore, EmbeddingManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Multi-Agent RAG Orchestrator API",
    description="Advanced multi-agent RAG system with specialized AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = None
active_workflows: Dict[str, Dict[str, Any]] = {}


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    workflow_type: Optional[str] = "comprehensive"
    config: Optional[Dict[str, Any]] = None


class DocumentUpload(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = "api_upload"


class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    message: str


class QueryResponse(BaseModel):
    workflow_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup"""
    global orchestrator
    
    try:
        logger.info("Initializing Multi-Agent RAG Orchestrator...")
        
        # Initialize core components
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore()
        
        # Initialize agents
        research_agent = ResearchAgent(vector_store, embedding_manager)
        analysis_agent = AnalysisAgent()
        writing_agent = WritingAgent()
        validation_agent = ValidationAgent()
        
        agents = {
            "research_agent": research_agent,
            "analysis_agent": analysis_agent,
            "writing_agent": writing_agent,
            "validation_agent": validation_agent
        }
        
        coordinator_agent = CoordinatorAgent(agents)
        agents["coordinator_agent"] = coordinator_agent
        
        # Initialize orchestrator
        orchestrator = MultiAgentRAGOrchestrator(
            agents=agents,
            embedding_manager=embedding_manager,
            vector_store=vector_store
        )
        
        # Add some sample documents
        await _add_sample_documents()
        
        logger.info("✅ Multi-Agent RAG Orchestrator initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise


async def _add_sample_documents():
    """Add sample documents to the knowledge base"""
    sample_documents = [
        {
            "content": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.",
            "metadata": {"source": "ai_overview.txt", "topic": "artificial_intelligence", "type": "educational"},
            "id": "doc_ai_overview"
        },
        {
            "content": "Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "metadata": {"source": "ml_intro.txt", "topic": "machine_learning", "type": "educational"},
            "id": "doc_ml_intro"
        },
        {
            "content": "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
            "metadata": {"source": "nlp_basics.txt", "topic": "nlp", "type": "educational"},
            "id": "doc_nlp_basics"
        }
    ]
    
    try:
        embeddings = await orchestrator.embedding_manager.generate_embeddings(
            [doc["content"] for doc in sample_documents]
        )
        
        await orchestrator.vector_store.add_documents(sample_documents, embeddings)
        logger.info(f"Added {len(sample_documents)} sample documents to knowledge base")
        
    except Exception as e:
        logger.error(f"Failed to add sample documents: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Agent RAG Orchestrator API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "upload": "/upload",
            "status": "/status/{workflow_id}",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a query through the multi-agent workflow"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    workflow_id = str(uuid.uuid4())
    
    # Store workflow info
    active_workflows[workflow_id] = {
        "status": "processing",
        "query": request.query,
        "workflow_type": request.workflow_type,
        "created_at": asyncio.get_event_loop().time()
    }
    
    # Start background processing
    background_tasks.add_task(
        _process_workflow_background,
        workflow_id,
        request.query,
        request.workflow_type,
        request.config or {}
    )
    
    return QueryResponse(
        workflow_id=workflow_id,
        status="accepted",
        result=None,
        error=None
    )


async def _process_workflow_background(workflow_id: str, query: str, workflow_type: str, config: Dict[str, Any]):
    """Process workflow in background"""
    try:
        # Execute workflow
        result = await orchestrator.execute_workflow(
            query=query,
            workflow_type=workflow_type,
            config=config
        )
        
        # Update workflow status
        active_workflows[workflow_id].update({
            "status": "completed",
            "result": result,
            "completed_at": asyncio.get_event_loop().time()
        })
        
    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")
        active_workflows[workflow_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": asyncio.get_event_loop().time()
        })


@app.get("/status/{workflow_id}", response_model=QueryResponse)
async def get_workflow_status(workflow_id: str):
    """Get the status of a specific workflow"""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_info = active_workflows[workflow_id]
    
    response = QueryResponse(
        workflow_id=workflow_id,
        status=workflow_info["status"],
        result=workflow_info.get("result"),
        error=workflow_info.get("error")
    )
    
    # Calculate processing time if completed
    if "completed_at" in workflow_info and "created_at" in workflow_info:
        response.processing_time = workflow_info["completed_at"] - workflow_info["created_at"]
    
    return response


@app.post("/upload", response_model=WorkflowResponse)
async def upload_document(document: DocumentUpload):
    """Upload a document to the knowledge base"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Prepare document
        doc_data = {
            "content": document.content,
            "metadata": document.metadata or {},
            "source": document.source,
            "id": doc_id
        }
        
        # Generate embedding
        embedding = await orchestrator.embedding_manager.generate_embedding(document.content)
        
        # Add to vector store
        await orchestrator.vector_store.add_documents([doc_data], [embedding])
        
        return WorkflowResponse(
            workflow_id=doc_id,
            status="success",
            message=f"Document uploaded successfully with ID: {doc_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not orchestrator:
        return {"status": "unhealthy", "reason": "Orchestrator not initialized"}
    
    try:
        # Perform basic health checks
        vs_stats = await orchestrator.vector_store.get_statistics()
        em_health = await orchestrator.embedding_manager.health_check()
        
        # Check agent status
        agent_status = {}
        for agent_name, agent in orchestrator.agents.items():
            metrics = agent.get_metrics()
            agent_status[agent_name] = {
                "status": agent.get_state().status,
                "tasks_completed": metrics["tasks_completed"]
            }
        
        return {
            "status": "healthy",
            "components": {
                "vector_store": {
                    "status": "healthy",
                    "documents": vs_stats["total_documents"]
                },
                "embedding_manager": {
                    "status": em_health["status"]
                },
                "agents": agent_status
            },
            "active_workflows": len(active_workflows)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "reason": str(e)}


@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        # Gather statistics
        vs_stats = await orchestrator.vector_store.get_statistics()
        em_stats = await orchestrator.embedding_manager.get_statistics()
        
        # Agent metrics
        agent_metrics = {}
        for agent_name, agent in orchestrator.agents.items():
            agent_metrics[agent_name] = agent.get_metrics()
        
        # Workflow statistics
        workflow_stats = {
            "total_workflows": len(active_workflows),
            "completed": len([w for w in active_workflows.values() if w["status"] == "completed"]),
            "failed": len([w for w in active_workflows.values() if w["status"] == "failed"]),
            "processing": len([w for w in active_workflows.values() if w["status"] == "processing"])
        }
        
        return {
            "vector_store": vs_stats,
            "embedding_manager": em_stats,
            "agents": agent_metrics,
            "workflows": workflow_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")


@app.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow from active workflows"""
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    del active_workflows[workflow_id]
    
    return {"message": f"Workflow {workflow_id} deleted successfully"}


@app.get("/workflows")
async def list_workflows():
    """List all active workflows"""
    workflows = []
    
    for workflow_id, workflow_info in active_workflows.items():
        workflows.append({
            "workflow_id": workflow_id,
            "status": workflow_info["status"],
            "query": workflow_info["query"],
            "workflow_type": workflow_info["workflow_type"],
            "created_at": workflow_info["created_at"]
        })
    
    return {"workflows": workflows, "total": len(workflows)}


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Multi-Agent RAG Orchestrator API Server...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API Explorer: http://localhost:8000/redoc")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
