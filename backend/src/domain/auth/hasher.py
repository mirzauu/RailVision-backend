from abc import ABC, abstractmethod


class PasswordHasher(ABC):
    @abstractmethod
    def verify(self, plain: str, hashed: str) -> bool:
        pass

    @abstractmethod
    def hash(self, password: str) -> str:
        pass