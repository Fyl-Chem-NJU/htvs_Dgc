import sys
from ccdc import search
from ccdc import io
import collections
import tqdm
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import tqdm
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem, PandasTools, BRICS, MACCSkeys, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity, DiceSimilarity
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors3D
import pandas as pd
import numpy as np

transition_metals = (
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Ac', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'
)
non_metal_backbone = (
    'B', 'C', 'N', 'O', 'Si', 'P', 'S', 'As', 'Se' 
)
def search_for_diphosphine_ligand_complexes(output_filename):


    searcher = search.SubstructureSearch()
    searcher.settings.has_3d_coordinates = True

    query = search.QuerySubstructure()

    P1 = query.add_atom('P')
    P2 = query.add_atom('P')
    R1 = query.add_atom(non_metal_backbone)
    M1 = query.add_atom(transition_metals)
    M2 = query.add_atom(transition_metals)

    query.add_bond('Any', P1, R1)  # P1-R1
    query.add_bond('Any', R1, P2)  # R1-P2
    query.add_bond('Any', P1, M1)  # P1-M
    query.add_bond('Any', P2, M2)  # P2-M

    searcher.add_substructure(query)

    print("Starting search in CSD for transition metal complexes with diphosphine ligands...")
    try:
        hits = searcher.search()
        print(f"Search finished! A total of {len(hits)} matching structures were found.")
        if not hits:
            print("No matches found.")
            return

    except Exception as e:
        print(f"Detailed error: {e}", file=sys.stderr)
        print("Please ensure that CSD is properly installed and that your license is valid and supports API access.", file=sys.stderr)

    return hits

def extract_specific_diphosphine_ligand_optimized(hit, metal_symbols):

    matched_atoms = hit.match_atoms()
    p1_atom = matched_atoms[0]
    p2_atom = matched_atoms[1]

    atoms_to_keep = set()
    queue = collections.deque([p1_atom, p2_atom])
    visited = {p1_atom, p2_atom}
    while queue:
        current_atom = queue.popleft()
        atoms_to_keep.add(current_atom)
        for neighbour in current_atom.neighbours:
            if neighbour not in visited and neighbour.atomic_symbol not in transition_metals:
                visited.add(neighbour)
                queue.append(neighbour)

    original_molecule = hit.entry.molecule
    atoms_no_keep = [atom for atom in original_molecule.atoms if atom not in atoms_to_keep]
    return original_molecule, atoms_no_keep

hits = search_for_diphosphine_ligand_complexes('test.mol')

data_records = []
seen_inchis = set()
error_count_identifier = 0
error_count_trans = 0

for hit in tqdm.tqdm(hits, desc="Processing hits"):
    identifier = hit.entry.identifier
    try:
        ligand, atoms_no_keep = extract_specific_diphosphine_ligand_optimized(hit, transition_metals)
        ligand.remove_atoms(atoms_no_keep)
        if ligand is None or not ligand.atoms:
            error_count_identifier += 1
            continue
    except Exception as e:
        error_count_identifier += 1
        continue
    try:
        hit_smiles = ligand.smiles
        if not hit_smiles:
            error_count_trans += 1
            continue
            
        mol = Chem.MolFromSmiles(hit_smiles)
        if mol is None:
            error_count_trans += 1
            continue
        inchi_tmp = Chem.MolToInchi(mol)
    except Exception as e:
        error_count_trans += 1
        continue
    data_records.append({
        'InChI': inchi_tmp,
        'SMILES': hit_smiles,
        'Identifier': identifier
    })
df_ligands = pd.DataFrame(data_records)

print(f"\nProcessing complete!")
print(f"Number of failed or empty ligand extractions: {error_count_identifier}")
print(f"Number of SMILES/InChI conversion failures: {error_count_trans}")
print(f"Final count of unique ligands obtained: {len(df_ligands)}.")
print(df_ligands.head())

df_ligands.to_csv('diphosphine_ligands.csv', index=False)

final_df_ligands = df_ligands.drop_duplicates(subset='InChI')


def has_radical(smiles: str) -> bool:

    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    if mol is None:
        print(f"Warning: SMILES '{smiles}' could not be parsed.")
        return False

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True
    return False

for smi in final_df_ligands['SMILES']:
    if has_radical(smi):
        print(f"SMILES '{smi}' contains radicals.")
        final_df_ligands = final_df_ligands.drop(final_df_ligands[final_df_ligands['SMILES'] == smi].index)
    else:
        print(f"SMILES '{smi}' does not contain radicals.")

identifier_list = final_df_ligands['Identifier'].tolist()

output_filename = "conquest_list.gcd"
try:
    with open(output_filename, 'w') as f:
        for identifier in identifier_list:
            f.write(identifier + '\n')
    print(f"Successfully created file: '{output_filename}'")
    print(f"File has {len(identifier_list)} Identifiers.")
except IOError as e:
    print(f"Error details: {e}")

with open('SMILES_list.smi', 'w') as f:
    for smi in final_df_ligands['SMILES']:
        f.write(smi + '\n')

with open('IDENTIFIER.ident', 'w') as f:
    for identifier in final_df_ligands['Identifier']:
        f.write(identifier + '\n')

SMI_IDEN_df = final_df_ligands[['SMILES', 'Identifier']]
SMI_IDEN_df.to_csv('SMILES_IDENTIFIER.csv', index=False)
