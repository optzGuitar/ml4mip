from enum import Enum

class BaseEnum(Enum):
    @classmethod
    def values(cls) -> list[str]:
        return [c.value for c in cls.items()]
    
    @classmethod
    def items(cls) -> list:
        return sorted([c for c in cls])
    