from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from tqdm import tqdm
from rdkit.Chem import Fragments
import pandas as pd

def is_nonorganic(fragment: "RDKit Mol object") -> bool:
    """Return true if fragment contains at least one carbon atom.
    :param fragment: The fragment as an RDKit Mol object.
    """
    # adapted from MolVS functiopn is_organic!!
    # TODO: Consider a different definition?
    # Could allow only H, C, N, O, S, P, F, Cl, Br, I
    for a in fragment.GetAtoms():
        if a.GetAtomicNum() == 6:
            return False
    return True

def contains_nonorg(fragment: "RDKit Mol object") -> str:
    # organic: H, C, N, O, P, S, F, Cl, Br, I
    """Return "Yes" if fragment does not contain at least one carbon atom.
    :param fragment: The fragment as an RDKit Mol object.
    """
    for a in fragment.GetAtoms():
        if a.GetAtomicNum() not in [1, 6, 7, 8, 15, 16, 9, 17, 35, 53]:
            return "Yes"
    return "No"

def standardize_compounds(smi_list: "List of SMILES strings", cids: list, stereo: str = "Clean") -> pd.DataFrame:
    """Return pandas dataframe containing a column with a note whether the molecule has been standardized
     (options: Failed at converting smiles to RDKit Mol Object, Standardized, Got empty molecule, Failed at disconnect,
     Failed at normalize, Failed at neutralising, Failed at stereochem remove, Failed at stereochem),
     index of input mol object, the latest standardized version of mol object, mixture: yes/no,
     contains unusual elements (CHNOPSFClBrI): yes/no,
     whether the error occured in standardization.
    """
    r = SaltRemover()
    molecule_column = smi_list
    stand_mol_list = []
    errs = []
    mixture = "No"
    for index, smi in tqdm(zip(cids, smi_list)):
        try:
            mol = Chem.MolFromSmiles(smi)
        except ValueError as e:
            stand_mol_list.append(("Failed at converting smiles to RDKit Mol Object", index, None, "No", None, str(e)))
            continue
        if mol is None:
            stand_mol_list.append(("Got empty molecule", index, mol, "No", None, None))
            continue
        try:
            mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)  # Disconnect metals
        except ValueError as e:
            if len(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)) > 1:
                mixture = "Yes"
            stand_mol_list.append(("Failed at disconnect", index, None, mixture, None, str(e)))
            continue

        mol = r.StripMol(mol)

        # Check if we have multiple fragments present

        if len(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)) > 1:
            mixture = "Yes"
        else:
            mixture = "No"

        # Standardize fragments separately

        for i, frag in enumerate(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)):

            frag = r.StripMol(frag)
            if frag.GetNumAtoms() == 0:
                continue
            elif is_nonorganic(frag):
                continue
            else:
                nonorg = contains_nonorg(frag)

                try:
                    frag = rdMolStandardize.Normalize(frag)
                except ValueError as e:
                    stand_mol_list.append(("Failed at normalize", index, None, mixture, nonorg, str(e)))
                    continue
                try:
                    frag = rdMolStandardize.Uncharger().uncharge(frag)
                except ValueError as e:
                    stand_mol_list.append(("Failed at neutralising", index, None, mixture, nonorg, str(e)))
                    continue

                if stereo == "Remove":
                    try:
                        Chem.RemoveStereochemistry(frag)
                    except ValueError as e:
                        stand_mol_list.append(("Failed at stereochem remove", index, None, mixture, nonorg, str(e)))
                        continue

                if stereo == "Clean":
                    ''' 
                    From RDKit documentation:
                    Does the CIP stereochemistry assignment

                    for the molecule’s atoms (R/S) and double bond (Z/E). Chiral atoms will have a property ‘_CIPCode’ indicating their chiral code.

                    ARGUMENTS:

                    mol: the molecule to use
                    cleanIt: (optional) if provided, atoms with a chiral specifier that aren’t actually chiral (e.g. atoms with duplicate substituents or only 2 substituents, etc.) will have their chiral code set to CHI_UNSPECIFIED. Bonds with STEREOCIS/STEREOTRANS specified that have duplicate substituents based upon the CIP atom ranks will be marked STEREONONE.
                    force: (optional) causes the calculation to be repeated, even if it has already been done
                    flagPossibleStereoCenters (optional) set the _ChiralityPossible property on atoms that are possible stereocenters

                    '''
                    try:
                        Chem.AssignStereochemistry(frag, force=True, cleanIt=True)
                    except ValueError as e:
                        stand_mol_list.append(("Failed at stereochem", index, None, mixture, nonorg, str(e)))
                        continue

            stand_mol_list.append(("Standardized", index, frag, mixture, nonorg, None))
    df_std = pd.DataFrame(data=stand_mol_list, columns=['Comment', 'cid', 'molecule', 'mixture', 'nonorg', 'error'])
    return df_std

def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol