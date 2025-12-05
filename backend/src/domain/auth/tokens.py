from abc import ABC, abstractmethod
from typing import Optional, Dict


class TokenProvider(ABC):
    @abstractmethod
    def create(self, subject: str) -> str:
        pass

    @abstractmethod
    def decode(self, token: str) -> Optional[Dict]:
        pass