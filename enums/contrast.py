from enums.base_enum import BaseEnum


class Contrasts(BaseEnum):
    flair = "FLAIR"
    t1w = "T1"
    t1ce = "T1CE"
    t2w = "T2"


class ClassificationContrasts(BaseEnum):
    flair = "FLAIR"
    t1w = "T1w"
    t1ce = "T1wCE"
    t2w = "T2w"
