from src.infrastructure.database.repositories.user_repository import UserRepository
from src.domain.auth.hasher import PasswordHasher
from src.domain.auth.tokens import TokenProvider
from src.infrastructure.database.models import User
from fastapi import HTTPException, status

class AuthService:
    def __init__(self, user_repo: UserRepository, hasher: PasswordHasher, token_provider: TokenProvider):
        self.user_repo = user_repo
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
        
        hashed_password = self.hasher.hash(password)
        new_user = User(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            org_id=org_id
        )
        return self.user_repo.create(new_user)

    def create_token_for_user(self, user: User):
        access_token = self.token_provider.create(subject=user.email)
        return {"access_token": access_token, "token_type": "bearer"}
