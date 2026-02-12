from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime
from decimal import Decimal

# Extraction Schemas (for LLM)
class ExtractedPipeline(BaseModel):
    arr_potential_cad: Optional[float] = Field(None, description="ARR Potential in CAD")
    status: Optional[str] = Field(None, description="Status: Target, Engaged, Pilot, Negotiation, Closed-Won, Closed-Lost")

class ExtractedAccount(BaseModel):
    account_name: str = Field(..., description="Name of the account/organization")
    segment: Optional[str] = Field(None, description="Segment like Passenger, Shortline, Class II")
    pipeline: Optional[ExtractedPipeline] = Field(None, description="Current pipeline status and potential")

class ExtractedPerformanceStudy(BaseModel):
    customer_name: Optional[str] = Field(None, description="Customer or Route name")
    metric_type: str = Field(..., description="Type of metric, e.g., 'fuel_savings_percent'")
    improvement_percent: float = Field(..., description="Improvement percentage, e.g., 7, 15, 25")
    measurement_period: Optional[str] = Field(None, description="Duration, e.g., '3 months'")
    methodology_notes: Optional[str] = Field(None, description="Context about how it was measured")

class ExtractedPartnerGeography(BaseModel):
    num_countries: int = Field(..., description="Number of reachable countries")
    regions: Optional[str] = Field(None, description="Regions covered")
    notes: Optional[str] = Field(None, description="Additional notes on geography")

class ExtractedPartner(BaseModel):
    partner_name: str = Field(..., description="Name of the partner")
    partnership_type: Optional[str] = Field(None, description="Type like Channel, OEM, Integrator")
    funding_amount_usd: Optional[float] = Field(None, description="Funding amount in USD")
    funding_notes: Optional[str] = Field(None, description="Purpose of funding")
    geography: Optional[ExtractedPartnerGeography] = Field(None, description="Geographic reach of this partner")

class CommercialMetricsExtraction(BaseModel):
    accounts: List[ExtractedAccount] = Field(default_factory=list)
    performance_studies: List[ExtractedPerformanceStudy] = Field(default_factory=list)
    partners: List[ExtractedPartner] = Field(default_factory=list)

# API Response Schemas
class PipelineResponse(BaseModel):
    arr_potential_cad: Optional[Decimal]
    status: Optional[str]
    snapshot_date: date

    class Config:
        from_attributes = True

class AccountResponse(BaseModel):
    id: str
    account_name: str
    segment: Optional[str]
    is_strategic_logo: bool
    pipelines: List[PipelineResponse] = []

    class Config:
        from_attributes = True

class PerformanceStudyResponse(BaseModel):
    id: str
    customer_name: Optional[str]
    metric_type: str
    improvement_percent: Decimal
    measurement_period: Optional[str]
    methodology_notes: Optional[str]

    class Config:
        from_attributes = True

class PartnerGeographyResponse(BaseModel):
    num_countries: Optional[int]
    regions: Optional[str]
    notes: Optional[str]

    class Config:
        from_attributes = True

class PartnerResponse(BaseModel):
    id: str
    partner_name: str
    partnership_type: Optional[str]
    funding_amount_usd: Optional[Decimal]
    funding_notes: Optional[str]
    geography: Optional[PartnerGeographyResponse]

    class Config:
        from_attributes = True

class CommercialMetricsResponse(BaseModel):
    accounts: List[AccountResponse]
    performance_studies: List[PerformanceStudyResponse]
    partners: List[PartnerResponse]
    last_updated: datetime = Field(default_factory=datetime.now)
