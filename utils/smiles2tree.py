from rdkit import Chem
import json
from collections import defaultdict
from enum import Enum
from utils.config import  ChiralityEnum, DegreeEnum, FormalChargeEnum, NumHsEnum, NumberRadicalElectronsEnum, HybridizationEnum, IsAromaticEnum, InRingEnum, BondStereoEnum, IsConjugatedEnum

# bond_type_mapping = {
#     "SINGLE": 1,
#     "DOUBLE": 2,
#     "TRIPLE": 3,
# }

def smiles_to_tree(atom_format, bond_format, atom_type_enum, smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    # Chem.Kekulize(mol, clearAromaticFlags=True)
    atom_idx_to_atom = {}
    visited_atoms = set()
    queue = []
    added_bonds = set([])
    neighbor_dict = defaultdict(list)

    # Initialize with the first atom
    for atom in mol.GetAtoms():
        if atom.GetIdx() == 0:  # Start with the first atom
            # print(f"Chirality: {str(atom.GetChiralTag())}, Degree: {atom.GetTotalDegree()}, Formal Charge: {atom.GetFormalCharge()}, Num Hs: {atom.GetTotalNumHs()}, Num Radical Electrons: {atom.GetNumRadicalElectrons()}, Hybridization: {str(atom.GetHybridization())}, Is Aromatic: {atom.GetIsAromatic()}, In Ring: {atom.IsInRing()}")
            atom_molecule = atom_format(atom_id=atom.GetIdx(), atom_type=atom.GetSymbol(), chirality=str(atom.GetChiralTag()), degree=str(atom.GetTotalDegree()), formal_charge=str(atom.GetFormalCharge()), num_hydrogens=str(atom.GetTotalNumHs()), num_radical_electrons=str(atom.GetNumRadicalElectrons()), hybridization=str(atom.GetHybridization()), is_aromatic=str(atom.GetIsAromatic()), in_ring=str(atom.IsInRing()), bonds=[])
            atom_idx_to_atom[atom.GetIdx()] = atom_molecule
            queue.append((None, atom.GetIdx(), None, None, None))  # (parent_idx, current_idx, bond_type, bond_stereo, is_conjugated)
            break

    # Process the queue using BFS
    while queue:
        parent_idx, current_idx, bond_type, bond_stereo, is_conjugated = queue.pop(0)
        current_atom = atom_idx_to_atom[current_idx]
        if parent_idx is not None:
            if current_idx in visited_atoms:
                current_atom = atom_format(atom_id=current_idx, atom_type=mol.GetAtomWithIdx(current_idx).GetSymbol(), chirality=str(mol.GetAtomWithIdx(current_idx).GetChiralTag()), degree=str(mol.GetAtomWithIdx(current_idx).GetTotalDegree()), formal_charge=str(mol.GetAtomWithIdx(current_idx).GetFormalCharge()), num_hydrogens=str(mol.GetAtomWithIdx(current_idx).GetTotalNumHs()), num_radical_electrons=str(mol.GetAtomWithIdx(current_idx).GetNumRadicalElectrons()), hybridization=str(mol.GetAtomWithIdx(current_idx).GetHybridization()), is_aromatic=str(mol.GetAtomWithIdx(current_idx).GetIsAromatic()), in_ring=str(mol.GetAtomWithIdx(current_idx).IsInRing()), bonds=[])
            parent_atom = atom_idx_to_atom[parent_idx]
            # print(bond_type)
            bond = bond_format(atom=current_atom, bond_type=bond_type, bond_stereo=str(bond_stereo), is_conjugated=str(is_conjugated))
            parent_atom.bonds.append(bond)
            visited_atoms.add(current_idx)
            visited_atoms.add(parent_idx)
            neighbor_dict[parent_idx].append(current_idx)
            
        # Explore the neighbors
        for bond in mol.GetAtomWithIdx(current_idx).GetBonds():
            neighbor_idx = bond.GetOtherAtomIdx(current_idx)
            if (current_idx, neighbor_idx) not in added_bonds:
                neighbor_bond_type = (str(bond.GetBondType()).replace("BondType.", ""))
                # neighbor_bond_type = bond_type_mapping[str(bond.GetBondType()).replace("BondType.", "")]
                if neighbor_bond_type == "DATIVE":
                    neighbor_bond_type = "SINGLE"
                if neighbor_idx not in atom_idx_to_atom:
                    neighbor_atom = atom_format(atom_id=neighbor_idx, atom_type=mol.GetAtomWithIdx(neighbor_idx).GetSymbol(), chirality=str(mol.GetAtomWithIdx(neighbor_idx).GetChiralTag()), degree=str(mol.GetAtomWithIdx(neighbor_idx).GetTotalDegree()), formal_charge=str(mol.GetAtomWithIdx(neighbor_idx).GetFormalCharge()), num_hydrogens=str(mol.GetAtomWithIdx(neighbor_idx).GetTotalNumHs()), num_radical_electrons=str(mol.GetAtomWithIdx(neighbor_idx).GetNumRadicalElectrons()), hybridization=str(mol.GetAtomWithIdx(neighbor_idx).GetHybridization()), is_aromatic=str(mol.GetAtomWithIdx(neighbor_idx).GetIsAromatic()), in_ring=str(mol.GetAtomWithIdx(neighbor_idx).IsInRing()), bonds=[])
                    atom_idx_to_atom[neighbor_idx] = neighbor_atom
                queue.append((current_idx, neighbor_idx, neighbor_bond_type, bond.GetStereo(), bond.GetIsConjugated()))
                added_bonds.add((current_idx, neighbor_idx))
    
    # Sort the atoms based on their indices
    analysis = []
    neighbor_dict = dict(sorted(neighbor_dict.items()))
    for atom_idx, neighbors in neighbor_dict.items():
        analysis.append(f"Atom with id {atom_idx} has neighbor atoms with indices: {', '.join(map(str, neighbors))}")
    
    analysis_str = ". ".join(analysis)
    return atom_idx_to_atom[0], analysis_str

# When serializing, convert enum to its value
def custom_json_serializer(obj):
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError("Type not serializable")

# Example usage
if __name__ == "__main__":
    smiles = "C1CC1CO"  # Cyclopropane with a hydroxymethyl group
    molecule = smiles_to_tree(smiles)
    molecule_json = json.dumps(molecule.dict(), indent=0, default=custom_json_serializer)
    print("========================================")
    print(molecule_json)