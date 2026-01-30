import os
import pubchempy
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds

def get_iupac(smiles):
    """
    Get the inpuc name from the smiles
    """
    compounds = pubchempy.get_compounds(smiles, namespace='smiles')
    match = compounds[0]
    return match.iupac_name

# * read the side chain file and standardize the smiles
side_chain = pd.read_csv('side_chain.csv')
side_chain_list = []
for smi in side_chain['side_chain_smiles']:
    mol_tmp = Chem.MolFromSmiles(smi)
    smi_tmp = Chem.MolToSmiles(mol_tmp)
    side_chain_list.append(smi_tmp)

# * get the XYZ coordinates of the side chain and save
output_dir = 'side_data/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for smi in side_chain_list:
    mol_side = Chem.MolFromSmiles(smi)
    # * get the number of ImplicitValence of each atom
    side_nbond_tmp = []
    for atom in mol_side.GetAtoms():
        implicit_valence = atom.GetImplicitValence()
        side_nbond_tmp.append(implicit_valence)

    # * get the XYZ coordinates of the side chain
    mol_side = Chem.AddHs(mol_side)
    AllChem.EmbedMolecule(mol_side)
    AllChem.MMFFOptimizeMolecule(mol_side)
    XYZ = Chem.MolToXYZBlock(mol_side)

    mol = Chem.MolFromXYZBlock(XYZ)

    tmp_name = get_iupac(smi)
    tmp_name = tmp_name.replace(',', '')
    file_name = tmp_name
    Chem.MolToXYZFile(mol, output_dir + file_name + '.xyz')

import csv
from molSimplify.Scripts.addtodb import addtoldb, initialize_custom_database
from molSimplify.Classes.globalvars import globalvars
from molSimplify.Scripts.generator import startgen_pythonic

# * initialize the database
globs = globalvars()
globs.add_custom_path('/mnt/d/FYL_project/htvs_Dgc/4_virtual_data/database')
initialize_custom_database(globs)

# * add the side chain to the database
for xyz_file in ['side_data_point/' + f for f in os.listdir('side_data_point/') if f.endswith('.xyz')]:
    xyz_name = os.path.basename(xyz_file).split('.')[0]
    j = int(xyz_name[-1])
    file_name = xyz_name
    source_file = 'side_data_point/' + file_name + '.xyz'

    smimol = source_file
    sminame = file_name
    smident = 1
    smicat = str(j)
    smigrps = ''
    smictg = 'group_tag'
    ffopt = 'BA'
    smichg = 0
    error_message = addtoldb(smimol, sminame, smident, smicat, smigrps, smictg, ffopt, smichg)
    if error_message:
        print(error_message)
    else:
        print(f"Ligand {sminame} added successfully.")


globs = globalvars()
globs.add_custom_path('/mnt/d/FYL_project/htvs_Dgc/4_virtual_data/database')

# * read the side chain file
side_func_name_list = []
with open('side_func_names.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        side_func_name_list.append(row[0])

for i in range(1,7):
    for side_func in tqdm.tqdm(side_func_name_list):
        dir_path = dir_path = f'/mnt/d/FYL_project/htvs_Dgc/4_virtual_data/tmp/side_structure/frame_{i}_{side_func}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            for step_type in ['ts', 'cat']:
                if i != 6:
                    if step_type == 'cat':
                        ccatoms = '7,11,15,19'
                    else:
                        ccatoms = '18,22,26,30'
                elif i == 6:
                    if step_type == 'cat':
                        ccatoms = '19,23,27,31'
                    else:
                        ccatoms = '30,34,38,42'

                name = f'{step_type}_frame_{i}_{side_func}'
                core_path = os.path.join('/mnt/d/FYL_project/htvs_Dgc/4_virtual_data',
                            'template', f'{step_type}_temp', f'{step_type}_frame_{i}.xyz')

                input_dict = {
                    '-name': name,
                    '-core': core_path,
                    '-lig': side_func,
                    '-ligocc': '4',
                    '-ccatoms': ccatoms,
                    '-rundir': dir_path,
                    '-keepHs': 'no',
                    '-replig': '1',
                    '-ligalign': '1',
                    '-geo': 'di_metal',

                }
                argv = ['main.py', '-i', 'asdfasdfasdfasdf']
                write_files = True
                flag = True
                gui = False
                startgen_pythonic(input_dict, argv, write=write_files, flag=flag, gui=gui)
                os.remove(os.path.join(dir_path, name, name, 'jobscript'))
                os.remove(os.path.join(dir_path, name, name, 'terachem_input'))
                os.remove(os.path.join(dir_path, name, name, f"{name}.molinp"))
                os.remove(os.path.join(dir_path, name, name, f"{name}.report"))