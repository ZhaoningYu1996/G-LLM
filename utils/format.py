from pydantic import BaseModel
from typing import List
from utils.config import  ChiralityEnum, DegreeEnum, FormalChargeEnum, NumHsEnum, NumberRadicalElectronsEnum, HybridizationEnum, IsAromaticEnum, InRingEnum, BondStereoEnum, IsConjugatedEnum

def create_mol_format(atom_type_enum, bond_type_enum):
    class Atom(BaseModel):
        atom_id: int
        atom_type: str
        chirality: str
        degree: str
        formal_charge: str
        num_hydrogens: str
        num_radical_electrons: str
        hybridization: str
        is_aromatic: str
        in_ring: str
        bonds: List['Bond'] = []
    
    class Bond(BaseModel):
        atom: Atom
        bond_type: str
        bond_stereo: str
        is_conjugated: str

    Atom.model_rebuild()
    Bond.model_rebuild()
    
    return Atom, Bond

# class Atom_id(BaseModel):
#     atom_id: int
#     atom_name: AtomEnum
#     bonds: List['Bond_id'] = []

# class Bond_id(BaseModel):
#     atom: Atom_id
#     bond_type: BondTypeEnum

# class Molecule(BaseModel):
#     atom_name: AtomEnum
#     adjacency_atoms: List['Molecule'] = []
#     bond_types: List[BondTypeEnum] = []

# class Atom(BaseModel):
#     atom_name: AtomEnum
#     bonds: List['Bond'] = []

# class Bond(BaseModel):
#     atom: Atom
#     bond_type: BondTypeEnum
