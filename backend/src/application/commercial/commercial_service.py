import logging
from typing import List, Optional
from uuid import uuid4
from datetime import datetime, date

from sqlalchemy.orm import Session
from sqlalchemy import delete

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
        You are a data extraction specialist.
        Based on the provided Strategic Facts and Supporting Context, extract the commercial metrics into the specified structure.
        
        {enriched_context}
        
        Extract:
        - Accounts with ARR potential and status.
        - Performance studies with specific improvement percentages.
        - Partners with funding and geography.
        """

        try:
            extraction: CommercialMetricsExtraction = await self.provider.call_llm_with_structured_output(
                messages=[{"role": "user", "content": prompt}],
                output_schema=CommercialMetricsExtraction,
                config_type="inference"  # or 'reasoning' if available/needed
            )
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            raise ValueError("Failed to extract structured data from Knowledge Base")

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
