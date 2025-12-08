from src.infrastructure.database.repositories.user_repository import UserRepository
from src.infrastructure.database.repositories.role_repository import RoleRepository
from src.domain.auth.hasher import PasswordHasher
from src.domain.auth.tokens import TokenProvider
from src.infrastructure.database.models import User
from fastapi import HTTPException, status

from src.infrastructure.database.repositories.org_repository import OrganizationRepository
from src.infrastructure.database.models import Organization

class AuthService:
    def __init__(self, user_repo: UserRepository, role_repo: RoleRepository, org_repo: OrganizationRepository, hasher: PasswordHasher, token_provider: TokenProvider):
        self.user_repo = user_repo
        self.role_repo = role_repo
        self.org_repo = org_repo
        self.hasher = hasher
        self.token_provider = token_provider

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
