from src.infrastructure.database.repositories.user_repository import UserRepository
from src.infrastructure.database.repositories.role_repository import RoleRepository
from src.domain.auth.hasher import PasswordHasher
from src.domain.auth.tokens import TokenProvider
from src.infrastructure.database.models import User
from fastapi import HTTPException, status

from src.infrastructure.database.repositories.org_repository import OrganizationRepository
from src.infrastructure.database.models import Organization
from datetime import datetime, timezone

from src.infrastructure.email.email_service import EmailService
import random
import string
from datetime import datetime, timezone, timedelta
from src.infrastructure.database.models import PasswordReset

class AuthService:
    def __init__(self, user_repo: UserRepository, role_repo: RoleRepository, org_repo: OrganizationRepository, hasher: PasswordHasher, token_provider: TokenProvider, email_service: EmailService = None):
        self.user_repo = user_repo
        self.role_repo = role_repo
        self.org_repo = org_repo
        self.hasher = hasher
        self.token_provider = token_provider
        self.email_service = email_service or EmailService()

    def authenticate_user(self, email: str, password: str):
        user = self.user_repo.get_by_email(email)
        if not user:
            return None
        if not self.hasher.verify(password, user.hashed_password):
            return None
        return user

    def register_user(self, email: str, password: str, full_name: str, org_id: str = None):
        existing_user = self.user_repo.get_by_email(email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Determine Organization
        if org_id:
            # Validate org exists
            org = self.org_repo.get_by_id(org_id)
            if not org:
                raise HTTPException(status_code=400, detail="Invalid organization ID")
            # If joining existing org, assign default role (e.g. viewer or developer)
            # For simplicity, let's assign 'viewer'
            role = self.role_repo.get_by_name("viewer") or self.role_repo.get_default_role()
        else:
            # Create new Organization
            org = Organization(name=f"{full_name}'s Organization", slug="".join(c for c in full_name.lower() if c.isalnum()))
            org = self.org_repo.create(org)
            org_id = org.id
            # Creator is Admin
            role = self.role_repo.get_by_name("org_admin")
        
        if not role:
             raise HTTPException(status_code=500, detail="Role configuration error")
             
        hashed_password = self.hasher.hash(password)
        new_user = User(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            org_id=org_id,
            role_id=role.id
        )
        return self.user_repo.create(new_user)

    def create_token_for_user(self, user: User):
        access_token = self.token_provider.create(subject=user.email)
        return {"access_token": access_token, "token_type": "bearer"}

    def create_login_response(self, user: User):
        tok = self.create_token_for_user(user)
        user_payload = {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "org_id": user.org_id,
            "role_id": user.role_id,
            "status": str(user.status.value) if hasattr(user, "status") else "active",
            "avatar_url": user.avatar_url,
        }
        tok["user"] = user_payload
        return tok

    def mark_user_login(self, user: User):
        now = datetime.now(timezone.utc)
        user.last_login_at = now
        user.last_active_at = now
        self.user_repo.db.add(user)
        self.user_repo.db.commit()
        self.user_repo.db.refresh(user)

    async def request_password_reset(self, email: str):
        user = self.user_repo.get_by_email(email)
        if not user:
            # We don't want to reveal if a user exists or not for security reasons
            # but usually for internal tools it's fine. 
            # For now, let's just return success anyway.
            return True
        
        # Generate a 6-digit OTP
        otp = ''.join(random.choices(string.digits, k=6))
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        
        reset_obj = PasswordReset(
            user_id=user.id,
            otp=otp,
            expires_at=expires_at
        )
        self.user_repo.create_password_reset(reset_obj)
        
        # Send email
        await self.email_service.send_otp(email, otp)
        return True

    def reset_password(self, email: str, otp: str, new_password: str):
        user = self.user_repo.get_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        reset_obj = self.user_repo.get_password_reset(user.id, otp)
        if not reset_obj:
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")
        
        # Update password
        user.hashed_password = self.hasher.hash(new_password)
        reset_obj.is_used = True
        
        self.user_repo.db.add(user)
        self.user_repo.db.add(reset_obj)
        self.user_repo.db.commit()
        return True
