from enums.base_enum import BaseEnum

class Contrasts(BaseEnum):
    flair = "FLAIR"
    t1w = "T1w"
    t1wce = "T1wCE"
    t2w = "T2w"
    
    @classmethod
    def values(cls):
        return sorted([c.value for c in cls])