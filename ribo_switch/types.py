from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional

class Base(IntEnum):
    A = 0
    C = 1
    G = 2
    U = 3

class BasePair(IntEnum):
    AU = 0
    UA = 1
    CG = 2
    GC = 3
    GU = 4
    UG = 5
CANONICAL_PAIRS: set[tuple[Base, Base]] = {(Base.A, Base.U), (Base.U, Base.A), (Base.C, Base.G), (Base.G, Base.C), (Base.G, Base.U), (Base.U, Base.G)}
Energy = int

def energy_to_kcal(e: Energy) -> float:
    return e / 100.0

@dataclass
class Sequence:
    bases: list[Base]

    def __len__(self):
        return len(self.bases)

    def __str__(self):
        return ''.join((b.name for b in self.bases))

@dataclass
class Structure:
    length: int
    pair_table: list[int]
    pairs: list[tuple[int, int]]

@dataclass
class HairpinLoop:
    closing_pair: tuple[int, int]
    unpaired: list[int]

@dataclass
class StackLoop:
    outer_pair: tuple[int, int]
    inner_pair: tuple[int, int]

@dataclass
class InteriorLoop:
    outer_pair: tuple[int, int]
    inner_pair: tuple[int, int]
    left_unpaired: list[int]
    right_unpaired: list[int]

@dataclass
class BulgeLoop:
    outer_pair: tuple[int, int]
    inner_pair: tuple[int, int]
    unpaired: list[int]
    side: str

@dataclass
class MultiLoop:
    closing_pair: tuple[int, int]
    branches: list[tuple[int, int]]
    unpaired: list[int]

@dataclass
class ExternalLoop:
    unpaired: list[int]
    closing_pairs: list[tuple[int, int]]
LoopType = HairpinLoop | StackLoop | InteriorLoop | BulgeLoop | MultiLoop | ExternalLoop
