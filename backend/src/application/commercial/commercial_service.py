import logging
import re
from typing import List, Optional
from uuid import uuid4
from datetime import datetime, date

from sqlalchemy.orm import Session
from sqlalchemy import delete

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider

from src.infrastructure.database.models.commercial import (
    Account, AccountPipeline, PerformanceStudy, Partner, PartnerGeography
)
from src.api.v1.dashboard.commercial_schemas import (
    CommercialMetricsResponse, CommercialMetricsExtraction,
    AccountResponse, PipelineResponse, PerformanceStudyResponse,
    PartnerResponse, PartnerGeographyResponse
)
from src.application.reasoning.pipeline import context_enrich
from src.infrastructure.llm.provider_service import ProviderService
from src.application.tools.service import ToolService

logger = logging.getLogger(__name__)

class CommercialService:
    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id
        self.provider = ProviderService(user_id=user_id)

    async def refresh_metrics(self, org_id: str) -> CommercialMetricsResponse:
        """
        Refreshes commercial metrics from the Knowledge Base using LLM extraction.
        """
        query = (
            "Extract the following commercial information for RailVision:\n"
            "1. Commercial Progress: List of strategic accounts (e.g., G&W, Watco) with their ARR Potential and Pipeline Status.\n"
            "2. Product Results: Performance studies showing fuel savings (e.g., 7%, 15%, 25%) and methodology.\n"
            "3. Strategic Leverage: Partners (e.g., Loram), their funding amounts, and geographic reach (countries/regions)."
        )

        # 1. Retrieve Context
        try:
            # We use a broad query to get all relevant context
            enriched_context = await context_enrich(
                question=query,
                user_id=self.user_id
            )
        except Exception as e:
            logger.error(f"Failed to enrich context: {e}")
            raise ValueError("Failed to retrieve information from Knowledge Base")

        # 2. Extract Structured Data via LLM
        prompt = f"""
        Analyze the provided baseline context and use your tools to perform a comprehensive commercial extraction for RailVision.
        
        You must extract structured information regarding:
        1. Strategic Accounts (e.g., G&W, Watco, CSX, etc.) including their ARR Potential and Pipeline Status.
        2. Performance Studies showing fuel savings (e.g., 7%, 15%, 25%) and methodology notes.
        3. Strategic Partners (e.g., Loram, Wabtec, Alstom) including funding amounts and geographic reach.
        
        Here is the initial baseline context gathered so far:
        ━━━━━━━━━━━━━━━━━━━━━━
        {enriched_context}
        ━━━━━━━━━━━━━━━━━━━━━━
        Current Known Constraints & Inferences:
        - Our internal documents do not always contain a confirmed figure for total capital raised to date.
        - Current fundraise target is often cited as $3M (strategic round) across Partnership strategy docs.
        - Always distinguish between documented facts and strategic inferences. Let the structured schema capture confirmed data.
        
        Instructions:
        1. Use the `think` tool to map out the exact data points you have found and note any gaps.
        2. If you feel crucial commercial metrics concerning Accounts, Studies, or Partners are missing from the baseline context, you MUST use the `knowledge_base` tool to search for specific details (e.g. search for "Strategic Partners", "Account Pipeline", or "Performance Studies").
        3. Once you have all the facts and have reasoned about them using the `think` tool, return the requested commercial metrics strictly matching the output schema.
        """

        try:
            tool_service = ToolService(self.db, self.user_id)
            langchain_tools = tool_service.get_tools(["think", "knowledge_base"])
            final_tools = []
            for t in langchain_tools:
                clean_name = re.sub(r" ", "", t.name)
                func = t.coroutine if t.coroutine else t.func
                final_tools.append(Tool(func, name=clean_name, description=t.description))

            provider_config = self.provider.chat_config
            provider = provider_config.provider
            api_key = self.provider._get_api_key(provider_config.auth_provider)
            base_url = provider_config.base_url
            model_id = provider_config.model.split("/")[-1]
            
            if provider == "openai":
                model = OpenAIModel(model_name=model_id, provider=OpenAIProvider(api_key=api_key, base_url=base_url))
            elif provider == "anthropic":
                try:
                    from anthropic import AsyncAnthropic
                    client = AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=10)
                    model = AnthropicModel(model_name=model_id, provider=AnthropicProvider(anthropic_client=client))
                except ImportError:
                    model = AnthropicModel(model_name=model_id, provider=AnthropicProvider(api_key=api_key, base_url=base_url))
            else:
                model = OpenAIModel(model_name=model_id, provider=OpenAIProvider(api_key=api_key, base_url=base_url))

            agent = PydanticAgent(
                model=model,
                tools=final_tools,
                result_type=CommercialMetricsExtraction,
                retries=3,
                system_prompt="You are a highly analytical Senior Commercial Researcher at RailVision. Your primary responsibility is to extract, verify, and structure commercial metrics accurately. You operate with strict fact discipline. Always use the 'think' tool to plan and reason about your data extraction step-by-step before producing a final answer. If you need more information about RailVision's operations, use the 'knowledge_base' tool to search through the company's internal documents.",
            )
            
            result = await agent.run(prompt)
            extraction: CommercialMetricsExtraction = result.data
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            raise ValueError(f"Failed to extract structured data from Knowledge Base: {e}")

        # 3. Update Database
        # For simplicity in this 'refresh' logic, we might clear old data or update.
        # Given the requirements, let's upsert Accounts and Partners, and add new Pipeline snapshots.
        # But to ensure the dashboard reflects EXACTLY what the KB says (and remove stale entries), 
        # a full refresh strategy (delete for org & re-insert) is often cleaner for "synced" views, 
        # UNLESS we need historical pipeline tracking. 
        # The user schema has 'snapshot_date' for pipeline, implying history.
        # So: 
        # - Accounts: Update or Insert.
        # - Pipeline: Insert new snapshot.
        # - Studies: Replace (assuming KB is the source of truth).
        # - Partners: Replace.

        try:
            self._update_accounts_and_pipeline(org_id, extraction.accounts)
            self._update_studies(org_id, extraction.performance_studies)
            self._update_partners(org_id, extraction.partners)
            
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Database update failed: {e}")
            raise ValueError(f"Failed to save metrics to database: {e}")

        return self.get_metrics(org_id)

    def _update_accounts_and_pipeline(self, org_id: str, accounts_data: List):
        for acc_data in accounts_data:
            # Find or Create Account
            account = self.db.query(Account).filter(
                Account.org_id == org_id,
                Account.account_name == acc_data.account_name
            ).first()

            if not account:
                account = Account(
                    org_id=org_id,
                    account_name=acc_data.account_name,
                    segment=acc_data.segment,
                    is_strategic_logo=True, # Default assumption from extraction
                    source='KB_LLM_Extract'
                )
                self.db.add(account)
                self.db.flush() # get ID
            else:
                # Update fields
                if acc_data.segment:
                    account.segment = acc_data.segment
            
            # Add Pipeline Snapshot
            if acc_data.pipeline:
                pipeline = AccountPipeline(
                    account_id=account.id,
                    arr_potential_cad=acc_data.pipeline.arr_potential_cad,
                    status=acc_data.pipeline.status,
                    snapshot_date=date.today(),
                    source='KB_LLM_Extract'
                )
                self.db.add(pipeline)

    def _update_studies(self, org_id: str, studies_data: List):
        # For studies, we replace existing ones to avoid duplicates if they don't have unique IDs in KB
        # or we could try to deduplicate based on metric/value.
        # Let's delete old KB-sourced studies and re-insert.
        self.db.query(PerformanceStudy).filter(
            PerformanceStudy.org_id == org_id,
            PerformanceStudy.source == 'KB_LLM_Extract'
        ).delete()

        for study in studies_data:
            ps = PerformanceStudy(
                org_id=org_id,
                customer_name=study.customer_name,
                metric_type=study.metric_type,
                improvement_percent=study.improvement_percent,
                measurement_period=study.measurement_period,
                methodology_notes=study.methodology_notes,
                source='KB_LLM_Extract'
            )
            self.db.add(ps)

    def _update_partners(self, org_id: str, partners_data: List):
        # Similar replace strategy for partners
        self.db.query(Partner).filter(
            Partner.org_id == org_id,
            Partner.source == 'KB_LLM_Extract'
        ).delete()
        # Note: cascading delete should handle geographies, but let's be safe/explicit if needed.
        # SQLAlchemy cascade="all, delete-orphan" on relationship handles it.

        for p_data in partners_data:
            partner = Partner(
                org_id=org_id,
                partner_name=p_data.partner_name,
                partnership_type=p_data.partnership_type,
                funding_amount_usd=p_data.funding_amount_usd,
                funding_notes=p_data.funding_notes,
                source='KB_LLM_Extract'
            )
            self.db.add(partner)
            self.db.flush()

            if p_data.geography:
                geo = PartnerGeography(
                    partner_id=partner.id,
                    num_countries=p_data.geography.num_countries,
                    regions=p_data.geography.regions,
                    notes=p_data.geography.notes,
                    source='KB_LLM_Extract'
                )
                self.db.add(geo)

    def get_metrics(self, org_id: str) -> CommercialMetricsResponse:
        accounts = self.db.query(Account).filter(Account.org_id == org_id).all()
        studies = self.db.query(PerformanceStudy).filter(PerformanceStudy.org_id == org_id).all()
        partners = self.db.query(Partner).filter(Partner.org_id == org_id).all()

        return CommercialMetricsResponse(
            accounts=[AccountResponse.model_validate(a) for a in accounts],
            performance_studies=[PerformanceStudyResponse.model_validate(s) for s in studies],
            partners=[PartnerResponse.model_validate(p) for p in partners],
            last_updated=datetime.now() # This is transient, maybe should fetch latest updated_at from DB
        )
