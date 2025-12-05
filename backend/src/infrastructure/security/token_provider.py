from typing import Optional, Dict
from src.domain.auth.tokens import TokenProvider
from src.shared.security import create_access_token, decode_access_token


class JoseJwtTokenProvider(TokenProvider):
    def create(self, subject: str) -> str:
        return create_access_token(subject=subject)

    def decode(self, token: str) -> Optional[Dict]:
        return decode_access_token(token)