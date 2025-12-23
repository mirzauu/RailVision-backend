from typing import Optional, List
import logging
from pydantic import BaseModel, Field
from src.infrastructure.vector.retriever import retrieve_context
from src.application.reasoning.state_builder import build_state
from src.infrastructure.llm.provider_service import ProviderService
from src.infrastructure.graph.schema import ALLOWED_NODE_TYPES, ALLOWED_RELATIONSHIPS

logger = logging.getLogger(__name__)

class IntentSchema(BaseModel):
    intent: str = Field(..., description="The classified intent of the user question.")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")

async def classify_intent(question: str, user_id: str = "system") -> str:
    """
    Classifies the user intent using LLM, referencing the graph schema.
    """
    svc = ProviderService(user_id=user_id)
    
    prompt = f"""
    You are an intent classifier for a strategic reasoning system.
    
    The system uses a Knowledge Graph with the following schema:
    Node Types: {", ".join(ALLOWED_NODE_TYPES)}
    Relationships: {", ".join(ALLOWED_RELATIONSHIPS)}
    
    Analyze the user's question and determine the most likely intent.
    The intent should be a short, descriptive string (e.g., "market_analysis", "risk_assessment", "capability_check", "general_inquiry").
    
    Question: {question}
    """
    
    try:
        result = await svc.call_llm_with_structured_output(
            messages=[{"role": "user", "content": prompt}],
            output_schema=IntentSchema,
            config_type="inference"
        )
        return result.intent
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "general"

async def context_enrich(
    question: str,
    doc_id: Optional[str] = None,
    active_version: int = 1, 
    allowed_categories: Optional[List[str]] = None,
    user_id: str = "system"
) -> str:
    """
    Enriches the user query with Strategic State (Neo4j) and Supporting Context (Pinecone).
    """
    if allowed_categories is None:
        allowed_categories = ["market", "go_to_market", "pricing", "general"]

    intent = await classify_intent(question, user_id)
    logger.info(f"Intent classified as: {intent}")
    
    # 1. Retrieve Context from Pinecone (Supporting Context)
    # We do this FIRST to finding relevant entities (doc_ids) for Neo4j
    context_matches = []
    relevant_doc_ids = []
    
    try:
        # Pass doc_id if provided (specific filter), otherwise None (broad search)
        query_doc_id = doc_id if doc_id else None
        
        matches = retrieve_context(
            query=question,
            doc_id=query_doc_id,
            active_version=str(active_version) if active_version else None, 
            allowed_categories=allowed_categories,
            top_k=5
        )
        
        if matches:
            context_matches = matches
            # Extract doc_ids from the metadata of matched chunks
            # m['text'] contains the full metadata dict based on retriever implementation
            relevant_doc_ids = list(set([
                m['text'].get('doc_id') 
                for m in matches 
                if isinstance(m.get('text'), dict) and m['text'].get('doc_id')
            ]))
            
            # If explicit doc_id was passed, ensure it is in the list
            if doc_id and doc_id not in relevant_doc_ids:
                relevant_doc_ids.append(doc_id)
                
    except Exception as e:
        logger.error(f"Failed to retrieve context: {e}")

    # 2. Build Strategic State (Neo4j)
    # Filtered by relevant doc_ids found in Pinecone OR keyword match
    try:
        strategic_state = build_state(doc_ids=relevant_doc_ids, query_text=question)
        strategic_state_str = str(strategic_state)
    except Exception as e:
        logger.error(f"Failed to build strategic state: {e}")
        strategic_state_str = "No strategic state available."

    # 3. Format Context String for Prompt
    context_str = "No supporting context available."
    if context_matches:
        # Extract the actual text content from metadata, assuming key is 'text' or 'chunk_text'
        # Adjust based on your actual metadata schema. 
        # Usually metadata={'text': '...', 'doc_id': '...'}
        formatted_matches = []
        for m in context_matches:
            meta = m['text']
            # safely get text
            content = meta.get('text') or meta.get('chunk_text') or str(meta)
            formatted_matches.append(f"- {content} (Score: {m['score']:.2f})")
        
        context_str = "\n\n".join(formatted_matches)

    # 4. Build reasoning input
    agent_input = f"""
STRATEGIC FACTS (Neo4j – authoritative):
{strategic_state_str}

SUPPORTING CONTEXT (Pinecone – explanatory):
{context_str}

QUESTION:
{question}
"""
    return agent_input
