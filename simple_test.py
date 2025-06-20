import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.orchestrator import MultiAgentRAGOrchestrator, QueryRequest

async def simple_test():
    print("🔧 Creating orchestrator...")
    orchestrator = MultiAgentRAGOrchestrator()
    print("✅ Orchestrator created")
    
    print("🚀 Starting orchestrator...")
    await orchestrator.start()
    print("✅ Orchestrator started")
    
    print("📄 Adding test document...")
    await orchestrator.add_documents(["This is a test document"])
    print("✅ Document added")
    
    print("❓ Processing test query...")
    request = QueryRequest(query="What is in the test document?")
    result = await orchestrator.process_query(request)
    print(f"✅ Response: {result.response}")
    
    await orchestrator.stop()
    print("✅ Test completed")

if __name__ == "__main__":
    asyncio.run(simple_test())
