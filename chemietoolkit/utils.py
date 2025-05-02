import numpy as np
from PIL import Image
import cv2
import layoutparser as lp
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
import re
import copy

BOND_TO_INT = {
    "": 0,
    "single": 1,
    "double": 2, 
    "triple": 3, 
    "aromatic": 4, 
    "solid wedge": 5, 
    "dashed wedge": 6
}

RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'Rf', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar', 'Ar1', 'Ar2', 'Ari', "R'", 
                  '1*', '2*','3*', '4*','5*', '6*','7*', '8*','9*', '10*','11*', '12*','[a*]', '[b*]','[c*]', '[d*]']

RGROUP_SYMBOLS = RGROUP_SYMBOLS + [f'[{i}]' for i in RGROUP_SYMBOLS]

RGROUP_SMILES = ['[1*]', '[2*]','[3*]', '[4*]','[5*]', '[6*]','[7*]', '[8*]','[9*]', '[10*]','[11*]', '[12*]','[a*]', '[b*]','[c*]', '[d*]','*', '[Rf]']

def get_figures_from_pages(pages, pdfparser):
    figures = []
    for i in range(len(pages)):
        img = np.asarray(pages[i])
        layout = pdfparser.detect(img)
        blocks = lp.Layout([b for b in layout if b.type == "Figure"])
        for block in blocks:
            figure = Image.fromarray(block.crop_image(img))
            figures.append({
                'image': figure,
                'page': i
            })
    return figures

def clean_bbox_output(figures, bboxes):
    results = []
    cropped = []
    references = []
    for i, output in enumerate(bboxes):
        mol_bboxes = [elt['bbox'] for elt in output if elt['category'] == '[Mol]']
        mol_scores = [elt['score'] for elt in output if elt['category'] == '[Mol]']
        data = {}
        results.append(data)
        data['image'] = figures[i]
        data['molecules'] = []
        for bbox, score in zip(mol_bboxes, mol_scores):
            x1, y1, x2, y2 = bbox
            height, width, _ = figures[i].shape
            cropped_img = figures[i][int(y1*height):int(y2*height),int(x1*width):int(x2*width)]
            cur_mol = {
                'bbox': bbox,
                'score': score,
                'image': cropped_img,
                #'info': None,
            }
            cropped.append(cropped_img)
            data['molecules'].append(cur_mol)
            references.append(cur_mol)
    return results, cropped, references
    
def convert_to_pil(image):
    if type(image) == np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    return image

def convert_to_cv2(image):
    if type(image) != np.ndarray:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return image

def replace_rgroups_in_figure(figures, results, coref_results, molscribe, batch_size=16):
    pattern = re.compile('(?P<name>[RXY]\d?)[ ]*=[ ]*(?P<group>\w+)')
    for figure, result, corefs in zip(figures, results, coref_results):
        r_groups = []
        seen_r_groups = set()
        for bbox in corefs['bboxes']:
            if bbox['category'] == '[Idt]':
                for text in bbox['text']:
                    res = pattern.search(text)
                    if res is None:
                        continue
                    name = res.group('name')
                    group = res.group('group')
                    if (name, group) in seen_r_groups:
                        continue
                    seen_r_groups.add((name, group))
                    r_groups.append({name: res.group('group')})
        if r_groups and result['reactions']:
            seen_r_groups = set([pair[0] for pair in seen_r_groups])
            orig_reaction = result['reactions'][0]
            graphs = get_atoms_and_bonds(figure['figure']['image'], orig_reaction, molscribe, batch_size=batch_size)
            relevant_locs = {}
            for i, graph in enumerate(graphs):
                to_add = []
                for j, atom in enumerate(graph['chartok_coords']['symbols']):
                    if atom[1:-1] in seen_r_groups:
                        to_add.append((atom[1:-1], j))
                relevant_locs[i] = to_add

            for r_group in r_groups:
                reaction = get_replaced_reaction(orig_reaction, graphs, relevant_locs, r_group, molscribe)
                to_add ={
                    'reactants': reaction['reactants'][:],
                    'conditions': orig_reaction['conditions'][:],
                    'products': reaction['products'][:]
                }
                result['reactions'].append(to_add)
    return results

def process_tables(figures, results, molscribe, batch_size=16):
    r_group_pattern = re.compile(r'^(\w+-)?(?P<group>[\w-]+)( \(\w+\))?$')
    for figure, result in zip(figures, results):
        result['page'] = figure['page']
        if figure['table']['content'] is not None:
            content = figure['table']['content']
            if len(result['reactions']) > 1:
                print("Warning: multiple reactions detected for table")
            elif len(result['reactions']) == 0:
                continue
            orig_reaction = result['reactions'][0]
            graphs = get_atoms_and_bonds(figure['figure']['image'], orig_reaction, molscribe, batch_size=batch_size)
            relevant_locs = find_relevant_groups(graphs, content['columns'])
            conditions_to_extend = []
            for row in content['rows']:
                r_groups = {}
                expanded_conditions = orig_reaction['conditions'][:]
                replaced = False
                for col, entry in zip(content['columns'], row):
                    if col['tag'] != 'alkyl group':
                        expanded_conditions.append({
                            'category': '[Table]',
                            'text': entry['text'], 
                            'tag': col['tag'],
                            'header': col['text'],
                        })
                    else:
                        found = r_group_pattern.match(entry['text'])
                        if found is not None:
                            r_groups[col['text']] = found.group('group')
                            replaced = True
                reaction = get_replaced_reaction(orig_reaction, graphs, relevant_locs, r_groups, molscribe)  
                if replaced:
                    to_add = {
                        'reactants': reaction['reactants'][:],
                        'conditions': expanded_conditions,
                        'products': reaction['products'][:]
                    }
                    result['reactions'].append(to_add)
                else:
                    conditions_to_extend.append(expanded_conditions)
            orig_reaction['conditions'] = [orig_reaction['conditions']]
            orig_reaction['conditions'].extend(conditions_to_extend)
    return results


def get_atoms_and_bonds(image, reaction, molscribe, batch_size=16):
    image = convert_to_cv2(image)
    cropped_images = []
    results = []
    for key, molecules in reaction.items():
        for i, elt in enumerate(molecules):
            if type(elt) != dict or elt['category'] != '[Mol]':
                continue
            x1, y1, x2, y2 = elt['bbox']
            height, width, _ = image.shape
            cropped_images.append(image[int(y1*height):int(y2*height),int(x1*width):int(x2*width)])
            to_add = {
                'image': cropped_images[-1],
                'chartok_coords': {
                    'coords': [],
                    'symbols': [],
                },
                'edges': [],
                'key': (key, i)
            }
            results.append(to_add)
    outputs = molscribe.predict_images(cropped_images, return_atoms_bonds=True, batch_size=batch_size)
    for mol, result in zip(outputs, results):
        for atom in mol['atoms']:
            result['chartok_coords']['coords'].append((atom['x'], atom['y']))
            result['chartok_coords']['symbols'].append(atom['atom_symbol'])
        result['edges'] = [[0] * len(mol['atoms']) for _ in range(len(mol['atoms']))]
        for bond in mol['bonds']:
            i, j = bond['endpoint_atoms']
            result['edges'][i][j] = BOND_TO_INT[bond['bond_type']]
            result['edges'][j][i] = BOND_TO_INT[bond['bond_type']]
    return results

def find_relevant_groups(graphs, columns):
    results = {}
    r_groups = set([f"[{col['text']}]" for col in columns if col['tag'] == 'alkyl group'])
    for i, graph in enumerate(graphs):
        to_add = []
        for j, atom in enumerate(graph['chartok_coords']['symbols']):
            if atom in r_groups:
                to_add.append((atom[1:-1], j))
        results[i] = to_add
    return results

def get_replaced_reaction(orig_reaction, graphs, relevant_locs, mappings, molscribe):
    graph_copy = []
    for graph in graphs:
        graph_copy.append({
            'image': graph['image'],
            'chartok_coords': {
                'coords': graph['chartok_coords']['coords'][:],
                'symbols': graph['chartok_coords']['symbols'][:],
            },
            'edges': graph['edges'][:],
            'key': graph['key'],
        })
    for graph_idx, atoms in relevant_locs.items():
        for atom, atom_idx in atoms:
            if atom in mappings:
                graph_copy[graph_idx]['chartok_coords']['symbols'][atom_idx] = mappings[atom]
    reaction_copy = {}
    def append_copy(copy_list, entity):    
        if entity['category'] == '[Mol]':
            copy_list.append({
                k1: v1 for k1, v1 in entity.items()
            })
        else:
            copy_list.append(entity)

    for k, v in orig_reaction.items():
        reaction_copy[k] = []
        for entity in v:
            if type(entity) == list:
                sub_list = []
                for e in entity:
                    append_copy(sub_list, e)
                reaction_copy[k].append(sub_list)
            else:
                append_copy(reaction_copy[k], entity)

    for graph in graph_copy:
        output = molscribe.convert_graph_to_output([graph], [graph['image']])
        molecule = reaction_copy[graph['key'][0]][graph['key'][1]]
        molecule['smiles'] = output[0]['smiles']
        molecule['molfile'] = output[0]['molfile']
    return reaction_copy

def get_sites(tar, ref, ref_site = False):
    rdDepictor.Compute2DCoords(ref)
    rdDepictor.Compute2DCoords(tar)
    idx_pair = rdDepictor.GenerateDepictionMatching2DStructure(tar, ref)

    in_template = [i[1] for i in idx_pair]
    sites = []
    for i in range(tar.GetNumAtoms()):
        if i not in in_template:
            for j in tar.GetAtomWithIdx(i).GetNeighbors():
                if j.GetIdx() in in_template and j.GetIdx() not in sites:

                    if ref_site: sites.append(idx_pair[in_template.index(j.GetIdx())][0])
                    else: sites.append(idx_pair[in_template.index(j.GetIdx())][0])
    return sites

def get_atom_mapping(prod_mol, prod_smiles, r_sites_reversed = None):
    # returns prod_mol_to_query which is the mapping of atom indices in prod_mol to the atom indices of the molecule represented by prod_smiles
    prod_template_intermediate = Chem.MolToSmiles(prod_mol)
    prod_template = prod_smiles

    for r in RGROUP_SMILES:
        if r!='*' and r!='(*)':
            prod_template = prod_template.replace(r, '*')
            prod_template_intermediate = prod_template_intermediate.replace(r, '*')

    prod_template_intermediate_mol = Chem.MolFromSmiles(prod_template_intermediate)
    prod_template_mol = Chem.MolFromSmiles(prod_template)
    
    p = Chem.AdjustQueryParameters.NoAdjustments()
    p.makeDummiesQueries = True
    
    prod_template_mol_query = Chem.AdjustQueryProperties(prod_template_mol, p)
    prod_template_intermediate_mol_query = Chem.AdjustQueryProperties(prod_template_intermediate_mol, p)
    rdDepictor.Compute2DCoords(prod_mol)
    rdDepictor.Compute2DCoords(prod_template_mol_query)
    rdDepictor.Compute2DCoords(prod_template_intermediate_mol_query)
    idx_pair = rdDepictor.GenerateDepictionMatching2DStructure(prod_mol, prod_template_intermediate_mol_query)
    
    intermdiate_to_prod_mol = {a:b for a,b in idx_pair}
    prod_mol_to_intermediate = {b:a for a,b in idx_pair}
    
    
    #idx_pair_2 = rdDepictor.GenerateDepictionMatching2DStructure(prod_template_mol_query, prod_template_intermediate_mol_query)
    
    #intermediate_to_query = {a:b for a,b in idx_pair_2}
    #query_to_intermediate = {b:a for a,b in idx_pair_2}
    
    #prod_mol_to_query = {a:intermediate_to_query[prod_mol_to_intermediate[a]] for a in prod_mol_to_intermediate}

 
    substructs = prod_template_mol_query.GetSubstructMatches(prod_template_intermediate_mol_query, uniquify = False)

    #idx_pair_2 = rdDepictor.GenerateDepictionMatching2DStructure(prod_template_mol_query, prod_template_intermediate_mol_query)
    for substruct in substructs:
        
        
        intermediate_to_query = {a:b for a, b in enumerate(substruct)}
        query_to_intermediate = {intermediate_to_query[i]: i for i in intermediate_to_query}

        prod_mol_to_query = {a:intermediate_to_query[prod_mol_to_intermediate[a]] for a in prod_mol_to_intermediate}
        
        good_map = True
        for i in r_sites_reversed:
            if prod_template_mol_query.GetAtomWithIdx(prod_mol_to_query[i]).GetSymbol() not in RGROUP_SMILES:
                good_map = False
        if good_map:
            break

    return prod_mol_to_query, prod_template_mol_query

def clean_corefs(coref_results_dict, idx):
    label_pattern = rf'{re.escape(idx)}[a-zA-Z]+'
    #unclean_pattern = re.escape(idx) + r'\d(?![\d% ])'
    toreturn = {}
    for prod in coref_results_dict:
        has_good_label = False
        for parsed in coref_results_dict[prod]:
            if re.search(label_pattern, parsed):
                has_good_label = True
        if not has_good_label:
            for parsed in coref_results_dict[prod]:
                if idx+'1' in parsed:
                    coref_results_dict[prod].append(idx+'l')
                elif idx+'0' in parsed:
                    coref_results_dict[prod].append(idx+'o')
                elif idx+'5' in parsed:
                    coref_results_dict[prod].append(idx+'s')
                elif idx+'9' in parsed:
                    coref_results_dict[prod].append(idx+'g')



def expand_r_group_label_helper(res, coref_smiles_to_graphs, other_prod, molscribe):
    name = res.group('name')
    group = res.group('group')
    #print(other_prod)
    atoms = coref_smiles_to_graphs[other_prod]['atoms']
    bonds = coref_smiles_to_graphs[other_prod]['bonds']

    #print(atoms, bonds)

    graph = {
        'image': None,
        'chartok_coords': {
            'coords': [],
            'symbols': [],
        },
        'edges': [],
        'key': None
    }
    for atom in atoms:
        graph['chartok_coords']['coords'].append((atom['x'], atom['y']))
        graph['chartok_coords']['symbols'].append(atom['atom_symbol'])
    graph['edges'] = [[0] * len(atoms) for _ in range(len(atoms))]
    for bond in bonds:
        i, j = bond['endpoint_atoms']
        graph['edges'][i][j] = BOND_TO_INT[bond['bond_type']]
        graph['edges'][j][i] = BOND_TO_INT[bond['bond_type']]
    for i, symbol in enumerate(graph['chartok_coords']['symbols']):
        if symbol[1:-1] == name:
            graph['chartok_coords']['symbols'][i] = group

    #print(graph)
    o = molscribe.convert_graph_to_output([graph], [graph['image']])
    return Chem.MolFromSmiles(o[0]['smiles'])

def get_r_group_frags_and_substitute(other_prod_mol, query, reactant_mols, reactant_information, parsed, toreturn):
    prod_template_mol_query, r_sites_reversed_new, h_sites, num_r_groups = query
    # we get the substruct matches. note that we set uniquify to false since the order matters for our method
    substructs = other_prod_mol.GetSubstructMatches(prod_template_mol_query, uniquify = False)

    
    #for r in r_sites_reversed:
    #    print(prod_template_mol_query.GetAtomWithIdx(prod_mol_to_query[r]).GetSymbol())

    # for each substruct we create the mapping of the substruct onto the other_mol
    # delete all the molecules in other_mol correspond to the substruct
    # and check if they number of mol frags is equal to number of r groups
    # we do this to make sure we have the correct substruct
    if len(substructs) >= 1:
        for substruct in substructs:

            query_to_other = {a:b for a,b in enumerate(substruct)}
            other_to_query = {query_to_other[i]:i for i in query_to_other}

            editable = Chem.EditableMol(other_prod_mol)
            r_site_correspondence = []
            for r in r_sites_reversed_new:
                #get its id in substruct
                substruct_id = query_to_other[r]
                r_site_correspondence.append([substruct_id, r_sites_reversed_new[r]])

            for idx in tuple(sorted(substruct, reverse = True)):
                if idx not in [query_to_other[i] for i in r_sites_reversed_new]:
                    editable.RemoveAtom(idx)
                    for r_site in r_site_correspondence:
                        if idx < r_site[0]:
                            r_site[0]-=1
            other_prod_removed = editable.GetMol()

            if len(Chem.GetMolFrags(other_prod_removed, asMols = False)) == num_r_groups:
                break

        # need to compute the sites at which correspond to each r_site_reversed

        r_site_correspondence.sort(key = lambda x: x[0])


        f = []
        ff = []
        frags = Chem.GetMolFrags(other_prod_removed, asMols = True, frags = f, fragsMolAtomMapping = ff)

        # r_group_information maps r group name --> the fragment/molcule corresponding to the r group and the atom index it should be connected at
        r_group_information = {}
        #tosubtract = 0
        for idx, r_site in enumerate(r_site_correspondence):

            r_group_information[r_site[1]]= (frags[f[r_site[0]]], ff[f[r_site[0]]].index(r_site[0]))
            #tosubtract += len(ff[idx])
        for r_site in h_sites:
            r_group_information[r_site] = (Chem.MolFromSmiles('[H]'), 0)

        # now we modify all of the reactants according to the R groups we have found
        # for every reactant we disconnect its r group symbol, and connect it to the r group
        modify_reactants = copy.deepcopy(reactant_mols)
        modified_reactant_smiles = []
        for reactant_idx in reactant_information:
            if len(reactant_information[reactant_idx]) == 0:
                modified_reactant_smiles.append(Chem.MolToSmiles(modify_reactants[reactant_idx]))
            else:
                combined = reactant_mols[reactant_idx]
                if combined.GetNumAtoms() == 1:
                    r_group, _, _ = reactant_information[reactant_idx][0]
                    modified_reactant_smiles.append(Chem.MolToSmiles(r_group_information[r_group][0]))
                else:
                    for r_group, r_index, connect_index in reactant_information[reactant_idx]:
                        combined = Chem.CombineMols(combined, r_group_information[r_group][0])

                    editable = Chem.EditableMol(combined)
                    atomIdxAdder = reactant_mols[reactant_idx].GetNumAtoms()
                    for r_group, r_index, connect_index in reactant_information[reactant_idx]:
                        Chem.EditableMol.RemoveBond(editable, r_index, connect_index)
                        Chem.EditableMol.AddBond(editable, connect_index, atomIdxAdder + r_group_information[r_group][1], Chem.BondType.SINGLE)
                        atomIdxAdder += r_group_information[r_group][0].GetNumAtoms()
                    r_indices = [i[1] for i in reactant_information[reactant_idx]]

                    r_indices.sort(reverse = True)

                    for r_index in r_indices:
                        Chem.EditableMol.RemoveAtom(editable, r_index)

                    modified_reactant_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(editable.GetMol()))))

        toreturn.append((modified_reactant_smiles, [Chem.MolToSmiles(other_prod_mol)], parsed))
        return True 
    else:
        return False

def query_enumeration(prod_template_mol_query, r_sites_reversed_new, num_r_groups):
    subsets = generate_subsets(num_r_groups)
    
    toreturn = []
    
    for subset in subsets:
        r_sites_list = [[i, r_sites_reversed_new[i]] for i in r_sites_reversed_new]
        r_sites_list.sort(key = lambda x: x[0])
        to_edit = Chem.EditableMol(prod_template_mol_query)
        
        for entry in subset:
            pos = r_sites_list[entry][0]
            Chem.EditableMol.RemoveBond(to_edit, r_sites_list[entry][0], prod_template_mol_query.GetAtomWithIdx(r_sites_list[entry][0]).GetNeighbors()[0].GetIdx())
        for entry in subset:
            pos = r_sites_list[entry][0]
            Chem.EditableMol.RemoveAtom(to_edit, pos)
        
        edited = to_edit.GetMol()
        for entry in subset: 
            for i in range(entry + 1, num_r_groups):
                r_sites_list[i][0]-=1

        new_r_sites = {}
        new_h_sites = set()
        for i in range(num_r_groups):
            if i not in subset:
                new_r_sites[r_sites_list[i][0]] = r_sites_list[i][1]
            else:
                new_h_sites.add(r_sites_list[i][1])
        toreturn.append((edited, new_r_sites, new_h_sites, num_r_groups - len(subset)))
    return toreturn
 
def generate_subsets(n):
    def backtrack(start, subset):
        result.append(subset[:])
        for i in range(start, -1, -1):  # Iterate in reverse order
            subset.append(i)
            backtrack(i - 1, subset)
            subset.pop()

    result = []
    backtrack(n - 1, [])
    return sorted(result, key=lambda x: (-len(x), x), reverse=True)

def backout(results, coref_results, molscribe):

    toreturn = []

    if not results or not results[0]['reactions'] or not coref_results:
        return toreturn

    try:
        reactants = results[0]['reactions'][0]['reactants']
        products = [i['smiles'] for i in results[0]['reactions'][0]['products']]
        coref_results_dict = {coref_results[0]['bboxes'][coref[0]]['smiles']: coref_results[0]['bboxes'][coref[1]]['text']  for coref in coref_results[0]['corefs']}
        coref_smiles_to_graphs = {coref_results[0]['bboxes'][coref[0]]['smiles']: coref_results[0]['bboxes'][coref[0]]  for coref in coref_results[0]['corefs']}
        
        
        if len(products) == 1:
            if products[0] not in coref_results_dict:
                print("Warning: No Label Parsed")
                return
            product_labels = coref_results_dict[products[0]]
            prod = products[0]
            label_idx = product_labels[0]
            '''
            if len(product_labels) == 1:
                # get the coreference label of the product molecule
                label_idx = product_labels[0]
            else:
                print("Warning: Malformed Label Parsed.")
                return
            '''
        else:
            print("Warning: More than one product detected")
            return
        
        # format the regular expression for labels that correspond to the product label
        numbers = re.findall(r'\d+', label_idx)
        label_idx = numbers[0] if len(numbers) > 0 else ""
        label_pattern = rf'{re.escape(label_idx)}[a-zA-Z]+'
        

        prod_smiles = prod
        prod_mol = Chem.MolFromMolBlock(results[0]['reactions'][0]['products'][0]['molfile'])
        
        # identify the atom indices of the R groups in the product tempalte
        h_counter = 0
        r_sites = {}
        for idx, atom in enumerate(results[0]['reactions'][0]['products'][0]['atoms']):
            sym = atom['atom_symbol']
            if sym == '[H]':
                h_counter += 1
            if sym[0] == '[':
                sym = sym[1:-1]
                if sym[0] == 'R' and sym[1:].isdigit():
                    sym = sym[1:]+"*"
                sym = f'[{sym}]'
            if sym in RGROUP_SYMBOLS:
                if sym not in r_sites:
                    r_sites[sym] = [idx-h_counter]
                else:
                    r_sites[sym].append(idx-h_counter)
        
        r_sites_reversed = {}
        for sym in r_sites:
            for pos in r_sites[sym]:
                r_sites_reversed[pos] = sym
        
        num_r_groups = len(r_sites_reversed)

        #prepare the product template and get the associated mapping

        prod_mol_to_query, prod_template_mol_query = get_atom_mapping(prod_mol, prod_smiles, r_sites_reversed = r_sites_reversed)
        
        reactant_mols = []
        
        
        #--------------process the reactants-----------------
        
        reactant_information = {} #index of relevant reaction --> [[R group name, atom index of R group, atom index of R group connection], ...]
        
        for idx, reactant in enumerate(reactants):
            reactant_information[idx] = []
            reactant_mols.append(Chem.MolFromSmiles(reactant['smiles']))
            has_r = False

            r_sites_reactant = {}
            
            h_counter = 0

            for a_idx, atom in enumerate(reactant['atoms']):
                
                #go through all atoms and check if they are an R group, if so add it to reactant information
                sym = atom['atom_symbol']
                if sym == '[H]':
                    h_counter += 1
                if sym[0] == '[':
                    sym = sym[1:-1]
                    if sym[0] == 'R' and sym[1:].isdigit():
                        sym = sym[1:]+"*"
                    sym = f'[{sym}]'
                if sym in r_sites:
                    if reactant_mols[-1].GetNumAtoms()==1:
                        reactant_information[idx].append([sym, -1, -1])
                    else: 
                        has_r = True
                        reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                        reactant_information[idx].append([sym, a_idx-h_counter, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx-h_counter).GetNeighbors()][0]])
                        r_sites_reactant[sym] = a_idx-h_counter
                elif sym == '[1*]' and '[7*]' in r_sites:
                    if reactant_mols[-1].GetNumAtoms()==1:
                        reactant_information[idx].append(['[7*]', -1, -1])
                    else: 
                        has_r = True
                        reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                        reactant_information[idx].append(['[7*]', a_idx-h_counter, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx-h_counter).GetNeighbors()][0]])
                        r_sites_reactant['[7*]'] = a_idx-h_counter
                elif sym == '[7*]' and '[1*]' in r_sites:
                    if reactant_mols[-1].GetNumAtoms()==1:
                        reactant_information[idx].append(['[1*]', -1, -1])
                    else: 
                        has_r = True
                        reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                        reactant_information[idx].append(['[1*]', a_idx-h_counter, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx-h_counter).GetNeighbors()][0]])
                        r_sites_reactant['[1*]'] = a_idx-h_counter



                elif sym == '[1*]' and '[Rf]' in r_sites:
                    if reactant_mols[-1].GetNumAtoms()==1:
                        reactant_information[idx].append(['[Rf]', -1, -1])
                    else: 
                        has_r = True
                        reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                        reactant_information[idx].append(['[Rf]', a_idx-h_counter, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx-h_counter).GetNeighbors()][0]])
                        r_sites_reactant['[Rf]'] = a_idx-h_counter

                elif sym == '[Rf]' and '[1*]' in r_sites:
                    if reactant_mols[-1].GetNumAtoms()==1:
                        reactant_information[idx].append(['[1*]', -1, -1])
                    else: 
                        has_r = True
                        reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                        reactant_information[idx].append(['[1*]', a_idx-h_counter, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx-h_counter).GetNeighbors()][0]])
                        r_sites_reactant['[1*]'] = a_idx-h_counter
                

            r_sites_reversed_reactant = {r_sites_reactant[i]: i for i in r_sites_reactant}
            # if the reactant had r groups, we had to use the molecule generated from the MolBlock. 
            # but the molblock may have unexpanded elemeents that are not R groups
            # so we have to map back the r group indices in the molblock version to the full molecule generated by the smiles
            # and adjust the indices of the r groups accordingly
            if has_r:
                #get the mapping
                reactant_mol_to_query, _ = get_atom_mapping(reactant_mols[-1], reactant['smiles'], r_sites_reversed = r_sites_reversed_reactant)

                #make the adjustment
                for info in reactant_information[idx]:
                    info[1] = reactant_mol_to_query[info[1]]
                    info[2] = reactant_mol_to_query[info[2]]
                reactant_mols[-1] = Chem.MolFromSmiles(reactant['smiles'])

        #go through all the molecules in the coreference

        clean_corefs(coref_results_dict, label_idx)

        for other_prod in coref_results_dict:

            #check if they match the product label regex
            found_good_label = False
            for parsed in coref_results_dict[other_prod]:
                if re.search(label_pattern, parsed) and not found_good_label:
                    found_good_label = True
                    other_prod_mol = Chem.MolFromSmiles(other_prod)

                    if other_prod != prod_smiles and other_prod_mol is not None:

                        #check if there are R groups to be resolved in the target product
                        
                        all_other_prod_mols = []
    
                        r_group_sub_pattern = re.compile('(?P<name>[RXY]\d?)[ ]*=[ ]*(?P<group>\w+)')

                        for parsed_labels in coref_results_dict[other_prod]:
                            res = r_group_sub_pattern.search(parsed_labels)

                            if res is not None:
                                all_other_prod_mols.append((expand_r_group_label_helper(res, coref_smiles_to_graphs, other_prod, molscribe), parsed + parsed_labels))
                        
                        if len(all_other_prod_mols) == 0:
                            if other_prod_mol is not None:
                                all_other_prod_mols.append((other_prod_mol, parsed))

                        
                            
                        
                        for other_prod_mol, parsed in all_other_prod_mols:
                        
                            other_prod_frags = Chem.GetMolFrags(other_prod_mol, asMols = True)
                            
                            for other_prod_frag in other_prod_frags:
                                substructs = other_prod_frag.GetSubstructMatches(prod_template_mol_query, uniquify = False)
                                
                                if len(substructs)>0:
                                    other_prod_mol = other_prod_frag
                                    break
                            r_sites_reversed_new = {prod_mol_to_query[r]: r_sites_reversed[r] for r in r_sites_reversed}

                            queries = query_enumeration(prod_template_mol_query, r_sites_reversed_new, num_r_groups)

                            matched = False

                            for query in queries:
                                if not matched:
                                    try:
                                        matched = get_r_group_frags_and_substitute(other_prod_mol, query, reactant_mols, reactant_information, parsed, toreturn)
                                    except:
                                        pass

    except:
        pass                          
                            
         
    return toreturn


def backout_without_coref(results, coref_results, coref_results_dict, coref_smiles_to_graphs, molscribe):

    toreturn = []

    if not results or not results[0]['reactions'] or not coref_results:
        return toreturn

    try:
        reactants = results[0]['reactions'][0]['reactants']
        products = [i['smiles'] for i in results[0]['reactions'][0]['products']]
        coref_results_dict = coref_results_dict
        coref_smiles_to_graphs = coref_smiles_to_graphs
        
        
        if len(products) == 1:
            if products[0] not in coref_results_dict:
                print("Warning: No Label Parsed")
                return
            product_labels = coref_results_dict[products[0]]
            prod = products[0]
            label_idx = product_labels[0]
            '''
            if len(product_labels) == 1:
                # get the coreference label of the product molecule
                label_idx = product_labels[0]
            else:
                print("Warning: Malformed Label Parsed.")
                return
            '''
        else:
            print("Warning: More than one product detected")
            return
        
        # format the regular expression for labels that correspond to the product label
        numbers = re.findall(r'\d+', label_idx)
        label_idx = numbers[0] if len(numbers) > 0 else ""
        label_pattern = rf'{re.escape(label_idx)}[a-zA-Z]+'
        

        prod_smiles = prod
        prod_mol = Chem.MolFromMolBlock(results[0]['reactions'][0]['products'][0]['molfile'])
        
        # identify the atom indices of the R groups in the product tempalte
        h_counter = 0
        r_sites = {}
        for idx, atom in enumerate(results[0]['reactions'][0]['products'][0]['atoms']):
            sym = atom['atom_symbol']
            if sym == '[H]':
                h_counter += 1
            if sym[0] == '[':
                sym = sym[1:-1]
                if sym[0] == 'R' and sym[1:].isdigit():
                    sym = sym[1:]+"*"
                sym = f'[{sym}]'
            if sym in RGROUP_SYMBOLS:
                if sym not in r_sites:
                    r_sites[sym] = [idx-h_counter]
                else:
                    r_sites[sym].append(idx-h_counter)
        
        r_sites_reversed = {}
        for sym in r_sites:
            for pos in r_sites[sym]:
                r_sites_reversed[pos] = sym
        
        num_r_groups = len(r_sites_reversed)

        #prepare the product template and get the associated mapping

        prod_mol_to_query, prod_template_mol_query = get_atom_mapping(prod_mol, prod_smiles, r_sites_reversed = r_sites_reversed)
        
        reactant_mols = []
        
        
        #--------------process the reactants-----------------
        
        reactant_information = {} #index of relevant reaction --> [[R group name, atom index of R group, atom index of R group connection], ...]
        
        for idx, reactant in enumerate(reactants):
            reactant_information[idx] = []
            reactant_mols.append(Chem.MolFromSmiles(reactant['smiles']))
            has_r = False

            r_sites_reactant = {}
            
            h_counter = 0

            for a_idx, atom in enumerate(reactant['atoms']):
                
                #go through all atoms and check if they are an R group, if so add it to reactant information
                sym = atom['atom_symbol']
                if sym == '[H]':
                    h_counter += 1
                if sym[0] == '[':
                    sym = sym[1:-1]
                    if sym[0] == 'R' and sym[1:].isdigit():
                        sym = sym[1:]+"*"
                    sym = f'[{sym}]'
                if sym in r_sites:
                    if reactant_mols[-1].GetNumAtoms()==1:
                        reactant_information[idx].append([sym, -1, -1])
                    else: 
                        has_r = True
                        reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                        reactant_information[idx].append([sym, a_idx-h_counter, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx-h_counter).GetNeighbors()][0]])
                        r_sites_reactant[sym] = a_idx-h_counter
                elif sym == '[1*]' and '[7*]' in r_sites:
                    if reactant_mols[-1].GetNumAtoms()==1:
                        reactant_information[idx].append(['[7*]', -1, -1])
                    else: 
                        has_r = True
                        reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                        reactant_information[idx].append(['[7*]', a_idx-h_counter, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx-h_counter).GetNeighbors()][0]])
                        r_sites_reactant['[7*]'] = a_idx-h_counter
                elif sym == '[7*]' and '[1*]' in r_sites:
                    if reactant_mols[-1].GetNumAtoms()==1:
                        reactant_information[idx].append(['[1*]', -1, -1])
                    else: 
                        has_r = True
                        reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                        reactant_information[idx].append(['[1*]', a_idx-h_counter, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx-h_counter).GetNeighbors()][0]])
                        r_sites_reactant['[1*]'] = a_idx-h_counter
                
                elif sym == '[1*]' and '[Rf]' in r_sites:
                    if reactant_mols[-1].GetNumAtoms()==1:
                        reactant_information[idx].append(['[Rf]', -1, -1])
                    else: 
                        has_r = True
                        reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                        reactant_information[idx].append(['[Rf]', a_idx-h_counter, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx-h_counter).GetNeighbors()][0]])
                        r_sites_reactant['[Rf]'] = a_idx-h_counter

                elif sym == '[Rf]' and '[1*]' in r_sites:
                    if reactant_mols[-1].GetNumAtoms()==1:
                        reactant_information[idx].append(['[1*]', -1, -1])
                    else: 
                        has_r = True
                        reactant_mols[-1] = Chem.MolFromMolBlock(reactant['molfile'])
                        reactant_information[idx].append(['[1*]', a_idx-h_counter, [i.GetIdx() for i in reactant_mols[-1].GetAtomWithIdx(a_idx-h_counter).GetNeighbors()][0]])
                        r_sites_reactant['[1*]'] = a_idx-h_counter

            r_sites_reversed_reactant = {r_sites_reactant[i]: i for i in r_sites_reactant}
            # if the reactant had r groups, we had to use the molecule generated from the MolBlock. 
            # but the molblock may have unexpanded elemeents that are not R groups
            # so we have to map back the r group indices in the molblock version to the full molecule generated by the smiles
            # and adjust the indices of the r groups accordingly
            if has_r:
                #get the mapping
                reactant_mol_to_query, _ = get_atom_mapping(reactant_mols[-1], reactant['smiles'], r_sites_reversed = r_sites_reversed_reactant)

                #make the adjustment
                for info in reactant_information[idx]:
                    info[1] = reactant_mol_to_query[info[1]]
                    info[2] = reactant_mol_to_query[info[2]]
                reactant_mols[-1] = Chem.MolFromSmiles(reactant['smiles'])

        #go through all the molecules in the coreference

        clean_corefs(coref_results_dict, label_idx)

        for other_prod in coref_results_dict:

            #check if they match the product label regex
            found_good_label = False
            for parsed in coref_results_dict[other_prod]:
                if re.search(label_pattern, parsed) and not found_good_label:
                    found_good_label = True
                    other_prod_mol = Chem.MolFromSmiles(other_prod)

                    if other_prod != prod_smiles and other_prod_mol is not None:

                        #check if there are R groups to be resolved in the target product
                        
                        all_other_prod_mols = []
    
                        r_group_sub_pattern = re.compile('(?P<name>[RXY]\d?)[ ]*=[ ]*(?P<group>\w+)')

                        for parsed_labels in coref_results_dict[other_prod]:
                            res = r_group_sub_pattern.search(parsed_labels)

                            if res is not None:
                                all_other_prod_mols.append((expand_r_group_label_helper(res, coref_smiles_to_graphs, other_prod, molscribe), parsed + parsed_labels))
                        
                        if len(all_other_prod_mols) == 0:
                            if other_prod_mol is not None:
                                all_other_prod_mols.append((other_prod_mol, parsed))

                        
                            
                        
                        for other_prod_mol, parsed in all_other_prod_mols:
                        
                            other_prod_frags = Chem.GetMolFrags(other_prod_mol, asMols = True)
                            
                            for other_prod_frag in other_prod_frags:
                                substructs = other_prod_frag.GetSubstructMatches(prod_template_mol_query, uniquify = False)
                                
                                if len(substructs)>0:
                                    other_prod_mol = other_prod_frag
                                    break
                            r_sites_reversed_new = {prod_mol_to_query[r]: r_sites_reversed[r] for r in r_sites_reversed}

                            queries = query_enumeration(prod_template_mol_query, r_sites_reversed_new, num_r_groups)

                            matched = False

                            for query in queries:
                                if not matched:
                                    try:
                                        matched = get_r_group_frags_and_substitute(other_prod_mol, query, reactant_mols, reactant_information, parsed, toreturn)
                                    except:
                                        pass

    except:
        pass                          
                            
         
    return toreturn



def associate_corefs(results, results_coref):
    coref_smiles = {}
    idx_pattern = r'\b\d+[a-zA-Z]{0,2}\b'
    for result_coref in results_coref:
        bboxes, corefs = result_coref['bboxes'], result_coref['corefs']
        for coref in corefs:
            mol, idt = coref[0], coref[1]
            if len(bboxes[idt]['text']) > 0:
                for text in bboxes[idt]['text']:
                    matches = re.findall(idx_pattern, text)
                    for match in matches:
                        coref_smiles[match] = bboxes[mol]['smiles']

    for page in results:
        for reactions in page['reactions']:
            for reaction in reactions['reactions']:
                if 'Reactants' in reaction:
                    if isinstance(reaction['Reactants'], tuple):
                        if reaction['Reactants'][0] in coref_smiles:
                            reaction['Reactants'] = (f'{reaction["Reactants"][0]} ({coref_smiles[reaction["Reactants"][0]]})', reaction['Reactants'][1], reaction['Reactants'][2])
                    else:
                        for idx, compound in enumerate(reaction['Reactants']):
                            if compound[0] in coref_smiles:
                                reaction['Reactants'][idx] = (f'{compound[0]} ({coref_smiles[compound[0]]})', compound[1], compound[2])
                if 'Product' in reaction:
                    if isinstance(reaction['Product'], tuple):
                        if reaction['Product'][0] in coref_smiles:
                            reaction['Product'] = (f'{reaction["Product"][0]} ({coref_smiles[reaction["Product"][0]]})', reaction['Product'][1], reaction['Product'][2])
                    else:
                        for idx, compound in enumerate(reaction['Product']):
                            if compound[0] in coref_smiles:
                                reaction['Product'][idx] = (f'{compound[0]} ({coref_smiles[compound[0]]})', compound[1], compound[2])
    return results


def expand_reactions_with_backout(initial_results, results_coref, molscribe): 
    idx_pattern = r'^\d+[a-zA-Z]{0,2}$'
    for reactions, result_coref in zip(initial_results, results_coref):
        if not reactions['reactions']:
            continue
        try:
            backout_results = backout([reactions], [result_coref], molscribe)
        except Exception:
            continue
        conditions = reactions['reactions'][0]['conditions']
        idt_to_smiles = {}
        if not backout_results:
            continue

        for reactants, products, idt in backout_results:
            reactions['reactions'].append({
                'reactants': [{'category': '[Mol]', 'molfile': None, 'smiles': reactant} for reactant in reactants],
                'conditions': conditions[:],
                'products': [{'category': '[Mol]', 'molfile': None, 'smiles': product} for product in products]
             })
    return initial_results

