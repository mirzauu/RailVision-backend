from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from src.config.database import get_db
from src.application.auth.auth_service import AuthService
from src.infrastructure.database.repositories.user_repository import UserRepository
from src.infrastructure.database.repositories.role_repository import RoleRepository
from src.infrastructure.database.repositories.org_repository import OrganizationRepository
from src.api.v1.auth.schemas import UserCreate, Token, UserResponse, UserLogin
from src.infrastructure.security.password_hasher import PasslibPasswordHasher
from src.infrastructure.security.token_provider import JoseJwtTokenProvider

router = APIRouter()

def get_auth_service(db: Session = Depends(get_db)) -> AuthService:
    user_repo = UserRepository(db)
    role_repo = RoleRepository(db)
    org_repo = OrganizationRepository(db)
    hasher = PasslibPasswordHasher()
    tokens = JoseJwtTokenProvider()
    return AuthService(user_repo, role_repo, org_repo, hasher, tokens)

@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, auth_service: AuthService = Depends(get_auth_service)):
    return auth_service.register_user(
        email=user_in.email,
        password=user_in.password,
        full_name=user_in.full_name,
        org_id=user_in.org_id
    )

@router.post("/login", response_model=Token)
def login(user_in: UserLogin, auth_service: AuthService = Depends(get_auth_service)):
    user = auth_service.authenticate_user(user_in.email, user_in.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_service.create_token_for_user(user)

@router.post("/token", response_model=Token)
def token(form_data: OAuth2PasswordRequestForm = Depends(), auth_service: AuthService = Depends(get_auth_service)):
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_service.create_token_for_user(user)
