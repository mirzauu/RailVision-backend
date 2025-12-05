from src.domain.auth.hasher import PasswordHasher
from src.shared.security import verify_password, get_password_hash


class PasslibPasswordHasher(PasswordHasher):
    def verify(self, plain: str, hashed: str) -> bool:
        return verify_password(plain, hashed)

    def hash(self, password: str) -> str:
        return get_password_hash(password)