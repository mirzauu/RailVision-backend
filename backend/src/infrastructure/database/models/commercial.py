from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, DateTime, Text, Numeric, Date
from sqlalchemy.orm import relationship
from src.config.database import Base
from .mixins import UUIDMixin, TimestampMixin
from sqlalchemy.sql import func

class Account(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "accounts"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    account_name = Column(String(255), nullable=False)
    segment = Column(String(100))  # e.g., 'Passenger', 'Shortline'
    is_strategic_logo = Column(Boolean, default=True)
    source = Column(String(100), nullable=False, default='KB_direct')

    # Relationships
    pipelines = relationship("AccountPipeline", back_populates="account", cascade="all, delete-orphan")
    organization = relationship("Organization") # Assuming Organization model exists and is imported in __init__ or via string

class AccountPipeline(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "account_pipelines"

    account_id = Column(String, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    arr_potential_cad = Column(Numeric(14, 2))  # KB: 10M, 5M, 3.5M, 1.5M
    status = Column(String(50))           # Target / Pilot / etc.
    snapshot_date = Column(Date, nullable=False, default=func.current_date())
    source = Column(String(100), nullable=False, default='KB_direct')

    account = relationship("Account", back_populates="pipelines")

class PerformanceStudy(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "performance_studies"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    customer_name = Column(String(255))
    metric_type = Column(String(100))   # 'fuel_savings_percent'
    improvement_percent = Column(Numeric(5, 2))  # 7, 15, 25 from KB
    measurement_period = Column(String(100))
    methodology_notes = Column(Text)
    source = Column(String(100), nullable=False, default='KB_direct')
    
    organization = relationship("Organization")

class Partner(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "partners"

    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    partner_name = Column(String(255), nullable=False)
    partnership_type = Column(String(100))          # TBD
    funding_amount_usd = Column(Numeric(14, 2))
    funding_notes = Column(Text)
    source = Column(String(100), nullable=False, default='KB_direct')

    geography = relationship("PartnerGeography", back_populates="partner", uselist=False, cascade="all, delete-orphan")
    organization = relationship("Organization")

class PartnerGeography(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "partner_geographies"

    partner_id = Column(String, ForeignKey("partners.id", ondelete="CASCADE"), nullable=False, unique=True)
    num_countries = Column(Integer)           # e.g., 32
    regions = Column(Text)          # TBD
    notes = Column(Text)
    source = Column(String(100), nullable=False, default='KB_direct')

    partner = relationship("Partner", back_populates="geography")
