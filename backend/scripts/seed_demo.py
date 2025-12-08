import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config.database import SessionLocal
from src.infrastructure.database.models.organizations import Organization
from src.infrastructure.database.models.users import User, Role, UserStatus
from src.shared.security import get_password_hash

def main():
    s = SessionLocal()
    try:
        org = s.query(Organization).filter_by(slug="demo-org").first()
        if org is None:
            org = Organization(name="Demo Org", slug="demo-org", description="Demo organization")
            s.add(org)
            s.commit()
            s.refresh(org)

        role = s.query(Role).filter_by(name="admin").first()
        if role is None:
            role = Role(name="admin", display_name="Admin", description="Admin role", permissions=[], is_system_role=True, is_default=False)
            s.add(role)
            s.commit()
            s.refresh(role)

        user = s.query(User).filter_by(email="demo@example.com").first()
        if user is None:
            hashed = get_password_hash("password123")
            user = User(org_id=org.id, role_id=role.id, email="demo@example.com", hashed_password=hashed, full_name="Demo User", email_verified=True, status=UserStatus.ACTIVE)
            s.add(user)
            s.commit()
            s.refresh(user)

        print({"org_id": org.id, "role_id": role.id, "user_id": user.id})
    finally:
        s.close()

if __name__ == "__main__":
    main()
