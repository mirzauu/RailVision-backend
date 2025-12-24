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
    The intent should be one of the following: "market_analysis", "risk_assessment", "capability_check", "financial_inquiry", "general_inquiry".
    
    Question: {question}
    """
    
    try:
        result = await svc.call_llm_with_structured_output(
            messages=[{"role": "user", "content": prompt}],
            output_schema=IntentSchema,
            config_type="inference"
        )
        return result.intent.lower()
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return "general_inquiry"

# Category Mapping Configuration
ALL_CATEGORIES = [
    "product", "market", "pricing", "traction", 
    "technology", "risk", "team", "financials", "other"
]

INTENT_CATEGORY_MAP = {
    "market_analysis": ["market", "pricing", "traction", "product"],
    "risk_assessment": ["risk", "financials", "legal", "other"],
    "capability_check": ["technology", "product", "team"],
    "financial_inquiry": ["financials", "pricing", "market"],
    "general_inquiry": ALL_CATEGORIES
}

async def context_enrich(
    question: str,
    active_version: Optional[str] = None, 
    allowed_categories: Optional[List[str]] = None,
    user_id: str = "system"
) -> str:
    """
    Enriches the user query with Strategic State (Neo4j) and Supporting Context (Pinecone).
    """
    # 1. Classify Intent to optimize retrieval scope
    intent = await classify_intent(question, user_id)
    logger.info(f"Intent classified as: {intent}")

    if allowed_categories is None:
        # Use intent to narrow down categories if not explicitly provided
        allowed_categories = INTENT_CATEGORY_MAP.get(intent, ALL_CATEGORIES)
        logger.info(f"Derived allowed_categories from intent: {allowed_categories}")
    
    # 2. Retrieve Context from Pinecone (Supporting Context)
    # We do this FIRST to finding relevant entities (doc_ids) for Neo4j
    context_matches = []
    
    try:
        matches = retrieve_context(
            query=question,
            active_version=active_version, 
            allowed_categories=allowed_categories,
            top_k=5
        )
        
        if matches:
            context_matches = matches
            
    except Exception as e:
        logger.error(f"Failed to retrieve context: {e}")
    
    # 2. Build Strategic State (Neo4j)
    # Filtered by keyword match only (attachment scoping removed)
    try:
        strategic_state = build_state(query_text=question)
        import json
        # Pretty print for better LLM readability
        strategic_state_str = json.dumps(strategic_state, indent=2)
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
