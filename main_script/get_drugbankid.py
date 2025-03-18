from rdkit import Chem
from rdkit.Chem import AllChem

# Input SMILES
one = 'CC(=O)[C@@]1(O)CC[C@H]2[C@@H]3C[C@H](C)C4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@@]21C'
two = '[H][C@@]12CC[C@](O)(C(C)=O)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])C[C@H](C)C2=CC(=O)C=C[C@]12C'
# three = 

# Convert SMILES to a molecule object
def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    # Generate Morgan fingerprint (radius=2, 2048-bit vector)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    # Convert to a bit string or list
    bit_string = fp.ToBitString()
    # bit_list = list(fp)
    return bit_string

# print(get_fp(one)==get_fp(two))

from rdkit import Chem
import random

original_smiles = "CC(=O)[C@@]1(O)CC[C@H]2[C@@H]3C[C@H](C)C4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@@]21C"

mol = Chem.MolFromSmiles(original_smiles)
if mol is None:
    print("Invalid SMILES")
else:
    num_atoms = mol.GetNumAtoms()
    seen = set()
    # Generate multiple non-canonical SMILES
    for _ in range(1000):  # Adjust iterations for more coverage
        new_order = list(range(num_atoms))
        random.shuffle(new_order)
        randomized_mol = Chem.RenumberAtoms(mol, new_order)
        randomized_smiles = Chem.MolToSmiles(randomized_mol, canonical=False)
        if randomized_smiles not in seen:
            seen.add(randomized_smiles)
    
    # Output unique SMILES
    # for smi in seen:
    #     print(smi)
# print(get_fp(list(seen)[0])==get_fp(list(seen)[1]))

print(len(list(seen)))