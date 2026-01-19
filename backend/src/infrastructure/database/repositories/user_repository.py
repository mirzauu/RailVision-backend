from sqlalchemy.orm import Session
from src.infrastructure.database.models import User
from typing import Optional, List

class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def get_by_id(self, user_id: str) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()

    def create(self, user: User) -> User:
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_by_org(self, org_id: str) -> List[User]:
        return self.db.query(User).filter(User.org_id == org_id).all()

    def create_password_reset(self, reset_obj):
        self.db.add(reset_obj)
        self.db.commit()
        self.db.refresh(reset_obj)
        return reset_obj

    def get_password_reset(self, user_id: str, otp: str):
        from src.infrastructure.database.models import PasswordReset
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return self.db.query(PasswordReset).filter(
            PasswordReset.user_id == user_id,
            PasswordReset.otp == otp,
            PasswordReset.expires_at > now,
            PasswordReset.is_used == False
        ).first()
