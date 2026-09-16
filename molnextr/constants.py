from typing import List
import re


ORGANIC_SET = {'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'}

RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'Rf', 'X', 'Y', 'Z', 'X1', 'X2', 'X3', 'X4', 'Y1', 'Y2', 'Y3', 'Y4', 'Z1', 'Z2', 'Z3', 'Z4', 'Q', 'A', 'E', 'Ar', 'Ar1', 'Ar2', 'Ari', 'Ar3', 'Ar4','Ar5','Ar6','Ar7',"R'", 
                  '1*', '2*','3*', '4*','5*', '6*','7*', '8*','9*', '10*','11*', '12*','[a*]', '[b*]','[c*]', '[d*]',"EWG",'Nu']

PLACEHOLDER_ATOMS = ["Lv", "Lu", "Nd", "Yb", "At", "Fm", "Er"]


class Substitution(object):
    '''Define common substitutions for chemical shorthand'''
    def __init__(self, abbrvs, smarts, smiles, probability):
        assert type(abbrvs) is list
        self.abbrvs = abbrvs
        self.smarts = smarts
        self.smiles = smiles
        self.probability = probability


SUBSTITUTIONS: List[Substitution] = [
    Substitution(['NO2', 'O2N'], '[N+](=O)[O-]', "[N+](=O)[O-]", 0.5),
    Substitution(['N2'], '[N+]=[N-]', "[N+]=[N-]", 0),
    Substitution(['CHO', 'OHC'], '[CH1](=O)', "[CH1](=O)", 0.5),
    Substitution(['CO2Et', 'COOEt','EtO2C'], 'C(=O)[OH0;D2][CH2;D2][CH3]', "[C](=O)OCC", 0.5),

    Substitution(['OAc'], '[OH0;X2]C(=O)[CH3]', "[O]C(=O)C", 0.7),
    Substitution(['NHAc'], '[NH1;D2]C(=O)[CH3]', "[NH]C(=O)C", 0.7),
    Substitution(['Ac'], 'C(=O)[CH3]', "[C](=O)C", 0.1),

    Substitution(['OBz'], '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[O]C(=O)c1ccccc1", 0.7),  # Benzoyl
    Substitution(['Bz'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)c1ccccc1", 0.2),  # Benzoyl

    Substitution(['OBn'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[O]Cc1ccccc1", 0.7),  # Benzyl
    Substitution(['Bn'], '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[CH2]c1ccccc1", 0.2),  # Benzyl

    Substitution(['NHBoc'], '[NH1;D2]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['NBoc'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['Boc'], 'C(=O)OC([CH3])([CH3])[CH3]', "[C](=O)OC(C)(C)C", 0.2),

    


    Substitution(['Cbm'], 'C(=O)[NH2;D1]', "[C](=O)N", 0.2),
    Substitution(['Cbz'], 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[C](=O)OCc1ccccc1", 0.4),
    Substitution(['Cy'], '[CH1;X3]1[CH2][CH2][CH2][CH2][CH2]1', "[CH1]1CCCCC1", 0.3),
    Substitution(['OCy'], '[O]C1CCCCC1', "[O]C1CCCCC1", 0.5),  # Phenyl
    Substitution(['Fmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[C](=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['Mes'], '[cH0]1c([CH3])cc([CH3])cc([CH3])1', "[c]1c(C)cc(C)cc(C)1", 0.5),
    Substitution(['OMs'], '[OH0;D2]S(=O)(=O)[CH3]', "[O]S(=O)(=O)C", 0.7),
    Substitution(['Ms'], 'S(=O)(=O)[CH3]', "[S](=O)(=O)C", 0.2),
    Substitution(['Ph'], '[cH0]1[cH][cH][cH1][cH][cH]1', "[c]1ccccc1", 0.5),
    Substitution(['PMB'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[CH2]c1ccc(OC)cc1", 0.2),
    Substitution(['Py'], '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', "[c]1ncccc1", 0.1),
    Substitution(['SEM'], '[CH2;D2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]CSi(C)(C)C", 0.2),
    Substitution(['Suc'], 'C(=O)[CH2][CH2]C(=O)[OH]', "[C](=O)CCC(=O)O", 0.2),
    Substitution(['TBS'], '[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['OTBS'], 'O[Si](C)(C)C(C)(C)C', "O[Si](C)(C)C(C)(C)C", 0.5),   # was C(C)(C)CC (tert-amyl) by mistake
    Substitution(['Bpin', 'BPin', 'B(pin)', 'pinB', 'PinB', '(pin)B'], 'B1OC(C)(C)C(C)(C)O1', "B1OC(C)(C)C(C)(C)O1", 0.3),
    Substitution(['TBZ'], 'C(=S)[cH]1[cH][cH][cH1][cH][cH]1', "[C](=S)c1ccccc1", 0.2),
    Substitution(['OTf'], '[OH0;D2]S(=O)(=O)C(F)(F)F', "[O]S(=O)(=O)C(F)(F)F", 0.7),
    Substitution(['Tf'], 'S(=O)(=O)C(F)(F)F', "[S](=O)(=O)C(F)(F)F", 0.2),
    Substitution(['TFA'], 'C(=O)C(F)(F)F', "[C](=O)C(F)(F)F", 0.3),
    Substitution(['TMS',"SiR2","SiR23"], '[Si]([CH3])([CH3])[CH3]', "[Si](C)(C)C", 0.5),
    Substitution(['Ts'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0.6),  # Tos
    Substitution(['OTMS',"OSiR2","OSiR23"], 'O[Si](C)(C)C', "O[Si](C)(C)C", 0.5),
    Substitution(['OPO(OEt)2'], '[O]P(=O)(OCC)OCC', "[O]P(=O)(OCC)OCC", 0.2),  # Tos
    Substitution(['OPO(OMe)2'], '[O]P(=O)(OC)OC', "[O]P(=O)(OC)OC", 0.2),
    Substitution(['TBDPSO','OTBDPS'], '[O][Si](C(C)(C)C)(c1ccccc1)c1ccccc1', '[O][Si](C(C)(C)C)(c1ccccc1)c1ccccc1', 0.2),  # Tos
    Substitution(['SO2Ph'], '[S](=O)(=O)c1ccccc1', '[S](=O)(=O)c1ccccc1', 0.5),
    Substitution(['SO2Me'], '[S](=O)(=O)[CH3]', '[S](=O)(=O)C', 0.5),
    Substitution(['SO2Et'], '[S](=O)(=O)[CH2;D2][CH3]', '[S](=O)(=O)CC', 0.5),
    Substitution(['SO2iPr'], '[S](=O)(=O)[CH1;D3]([CH3])[CH3]', '[S](=O)(=O)C(C)C', 0.5),
    Substitution(['SO2tBu'], '[S](=O)(=O)[CH0]([CH3])([CH3])[CH3]', '[S](=O)(=O)C(C)(C)C', 0.5),
    Substitution(['Piv'], '[C](=O)C(C)(C)C', '[C](=O)C(C)(C)C', 0.5),
    Substitution(['PivO','OPiv'], '[O]C(=O)C(C)(C)C', "[O]C(=O)C(C)(C)C", 0.5),  # Phenyl



    # Alkyl chains

    Substitution(['OMe', 'MeO'], '[OH0;D2][CH3;D1]', "[O]C", 0.3),
    Substitution(['OAr'], '[O](*)', '[O](*)', 0.5),
    Substitution(['SMe', 'MeS'], '[SH0;D2][CH3;D1]', "[S]C", 0.3),
    Substitution(['NMe', 'MeN'], '[N;X3][CH3;D1]', "[NH]C", 0.3),
    Substitution(['Me'], '[CH3;D1]', "[CH3]", 0.1),
    Substitution(['OEt', 'EtO'], '[OH0;D2][CH2;D2][CH3]', "[O]CC", 0.5),
    Substitution(['Et', 'C2H5'], '[CH2;D2][CH3]', "[CH2]C", 0.3),
    Substitution(['Pr', 'nPr', 'n-Pr'], '[CH2;D2][CH2;D2][CH3]', "[CH2]CC", 0.3),
    Substitution(['Bu', 'nBu', 'n-Bu'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCC", 0.3),
    Substitution(['OPh', 'OPh'], '[O]c1ccccc1', "[O]c1ccccc1", 0.2),

    # Branched
    Substitution(['iPr', 'i-Pr'], '[CH1;D3]([CH3])[CH3]', "[CH1](C)C", 0.2),
    Substitution(['iPrO', 'i-PrO','OiPr'], '[OH0;D2][CH1;D3]([CH3])[CH3]', "[O]C(C)C", 0.2),
    Substitution(['iBu', 'i-Bu'], '[CH2;D2][CH1;D3]([CH3])[CH3]', "[CH2]C(C)C", 0.2),
    Substitution(['OiBu'], '[OH0;D2][CH2;D2][CH1;D3]([CH3])[CH3]', "[O]CC(C)C", 0.2),
    Substitution(['OtBu'], '[OH0;D2][CH0]([CH3])([CH3])[CH3]', "[O]C(C)(C)C", 0.6),
    Substitution(['tBu', 't-Bu'], '[CH0]([CH3])([CH3])[CH3]', "[C](C)(C)C", 0.3),
    Substitution(['CO2Me', 'MeO2C'], '[C](=O)OC', "[C](=O)OC", 0.3),
    Substitution(['MeO2CO', 'OCO2Me'], '[O]C(=O)OC', "[O]C(=O)OC", 0.3),
    Substitution(['ONa', 'NaO'], '[O-].[Na+]', "[O-].[Na+]", 0.3),

    # Other shorthands (MIGHT NOT WANT ALL OF THESE)
    Substitution(['CF3', 'F3C'], '[CH0;D4](F)(F)F', "[C](F)(F)F", 0.5),
    Substitution(['NCF3', 'F3CN'], '[N;X3][CH0;D4](F)(F)F', "[NH]C(F)(F)F", 0.5),
    Substitution(['OCF3', 'F3CO'], '[OH0;X2][CH0;D4](F)(F)F', "[O]C(F)(F)F", 0.5),
    Substitution(['CCl3'], '[CH0;D4](Cl)(Cl)Cl', "[C](Cl)(Cl)Cl", 0.5),
    Substitution(['CO2H', 'HO2C', 'COOH'], 'C(=O)[OH]', "[C](=O)O", 0.5),  # COOH
    Substitution(['CN', 'NC'], 'C#[ND1]', "[C]#N", 0.5),
    Substitution(['OCH3', 'H3CO','CH3O'], '[OH0;D2][CH3]', "[O]C", 0.4),
    Substitution(['SO3H'], 'S(=O)(=O)[OH]', "[S](=O)(=O)O", 0.4),


    ###MOLNEXTR
    Substitution(['C5H17','C5H11','C5H14'], 'CCCCC', "CCCCC", 0.0),
    Substitution(['C4H9','C4H10'], 'CCCC', "CCCC", 0.0),
    Substitution(['C3H7','C3H8'], 'CCC', "CCC", 0.0),
    Substitution(['C2H5','C2H6'], 'CC', "CC", 0.0),
    Substitution(['C11H23','C17H23'], 'CCCCCCCCCCC', "CCCCCCCCCCC", 0.0),
    Substitution(['Alyl','Allyl'], 'C=CC', "C=CC", 0.0),
    Substitution(['OAll','OAlI'], 'OCC=C', "OCC=C", 0.0),

    Substitution(['N3'], 'N=[N+]=[N-]', "N=[N+]=[N-]", 0.2),
    Substitution(['N2+'], 'N#[N+]', "N#[N+]", 0),
    Substitution(['N2'], '[N+]=[N-]', "[N+]=[N-]", 0),
    Substitution(['Tos','Tcs'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0),  # Tos
    Substitution(['OTBDMS'], '[OH0;D2][Si](C)(C)C(C)(C)C', "[O][Si](C)(C)C(C)(C)C", 0),  # TBDMS
    Substitution(['SP'], 'S[P]', "S[P]", 0), # Sulfenyl Phosphide
    Substitution(['CH3O'], '[OH0;D2][CH3]', "[O]C", 0),
    Substitution(['OCN','NCO'], 'N=C=O', "N=C=O", 0),
    Substitution(['SO2NH2'], 'S(N)(=O)=O', "S(N)(=O)=O", 0),
    Substitution(['NHCOtBu'], 'NC(=O)C(C)(C)C', "NC(=O)C(C)(C)C", 0),
    Substitution(['SPh'], 'Sc1ccccc1', "Sc1ccccc1", 0),
    Substitution(['EtOH'], '[CH2;D2][CH3;D1][OH0;D2]', "[CH2]CO", 0),  # Ethanol
    Substitution(['TBA'], '[CH3;D1][C;D4]([CH3;D1])([CH3;D1])[CH3;D1]', "[CH3]C(C)(C)C", 0),  # Tert-Butyl alcohol
    Substitution(['DMF'], 'CN(C)C=O', "CN(C)C=O", 0),  # Dimethylformamide
    Substitution(['DMSO'], 'CS(=O)C', "CS(=O)C", 0),  # Dimethyl sulfoxide
    Substitution(['THF'], 'C1CCCO1', "C1CCCO1", 0),  # Tetrahydrofuran
#Substitution(['C19H15'], 'CC1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3', 'CC1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3', 0.5),# Trityl
    ### complex substituents
    Substitution(['-ClO4'], "Cl([O-])(=O)(=O)=O", "Cl([O-])(=O)(=O)=O", 0),  
    Substitution(['-OTf'], "O=S(=O)([O-])C(F)(F)F", "O=S(=O)([O-])C(F)(F)F", 0),  
    Substitution(['-BF4','BF4'], 'F[B-](F)(F)F', "F[B-](F)(F)F", 0),  



    Substitution(['Me3SiO'], '[O][Si](C)(C)C', "[O][Si](C)(C)C", 0.2),
    ### acyl groups written as CO-R (not in the table they were parsed as C-O-R with a radical carbon)
    Substitution(['COMe', 'MeCO', 'C(O)Me'], 'C(=O)[CH3]', "[C](=O)C", 0.2),
    Substitution(['COEt', 'EtCO', 'C(O)Et'], 'C(=O)[CH2][CH3]', "[C](=O)CC", 0.2),
    Substitution(['COnPr', 'COPr', 'nPrCO', 'PrCO'], 'C(=O)[CH2][CH2][CH3]', "[C](=O)CCC", 0.2),
    Substitution(['COiPr', 'iPrCO'], 'C(=O)[CH1]([CH3])[CH3]', "[C](=O)C(C)C", 0.2),
    Substitution(['COtBu', 'tBuCO', 'Piv'], 'C(=O)C([CH3])([CH3])[CH3]', "[C](=O)C(C)(C)C", 0.2),
    Substitution(['COPh', 'PhCO'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)c1ccccc1", 0.2),
    Substitution(['COBn', 'BnCO'], 'C(=O)[CH2][cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)Cc1ccccc1", 0.2),
    Substitution(['COCy', 'CyCO'], 'C(=O)[CH1]1[CH2][CH2][CH2][CH2][CH2]1', "[C](=O)C1CCCCC1", 0.2),
    Substitution(['OPiv'], '[OH0;D2]C(=O)C([CH3])([CH3])[CH3]', "[O]C(=O)C(C)(C)C", 0.3),
    ### naphthyl, THP, N-protected amines, longer alkyls (were expanded to a wildcard)
    Substitution(['1-Npth', '1-Naph', 'Npth', 'Naph', '1-Np', 'a-Naph'], '[cH0]1[cH][cH][cH]c2[cH][cH][cH][cH]c12', "[c]1cccc2ccccc12", 0.3),
    Substitution(['2-Npth', '2-Naph', '2-Np', 'b-Naph'], '[cH0]1[cH][cH]c2[cH][cH][cH][cH]c2[cH]1', "[c]1ccc2ccccc2c1", 0.3),
    Substitution(['OTHP'], '[OH0;D2][CH1]1[CH2][CH2][CH2][CH2]O1', "[O]C1CCCCO1", 0.3),
    Substitution(['THP'], '[CH1]1[CH2][CH2][CH2][CH2]O1', "[CH]1CCCCO1", 0.2),
    Substitution(['NMeBoc', 'NBocMe', 'N(Me)Boc'], '[NH0;D3]([CH3])C(=O)OC([CH3])([CH3])[CH3]', "[N](C)C(=O)OC(C)(C)C", 0.3),
    Substitution(['NHCbz'], '[NH1;D2]C(=O)OC[cH0]1[cH][cH][cH][cH][cH]1', "[NH]C(=O)OCc1ccccc1", 0.3),
    Substitution(['Pent', 'nPent', 'n-Pent', 'Pen'], '[CH2;D2][CH2][CH2][CH2][CH3]', "[CH2]CCCC", 0.2),
    Substitution(['iPent', 'i-Pent', 'isoPent'], '[CH2;D2][CH2][CH1]([CH3])[CH3]', "[CH2]CC(C)C", 0.2),
    Substitution(['neoPent', 'neo-Pent'], '[CH2;D2]C([CH3])([CH3])[CH3]', "[CH2]C(C)(C)C", 0.2),
    Substitution(['Hept', 'nHept', 'n-Hept'], '[CH2;D2][CH2][CH2][CH2][CH2][CH2][CH3]', "[CH2]CCCCCC", 0.2),
    Substitution(['Hex', 'nHex', 'n-Hex'], '[CH2;D2][CH2][CH2][CH2][CH2][CH3]', "[CH2]CCCCC", 0.2),
    Substitution(['Oct', 'nOct', 'n-Oct'], '[CH2;D2][CH2][CH2][CH2][CH2][CH2][CH2][CH3]', "[CH2]CCCCCCC", 0.2),
    Substitution(['Non', 'nNon', 'n-Non'], '[CH2;D2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3]', "[CH2]CCCCCCCC", 0.2),
    Substitution(['Dec', 'nDec', 'n-Dec'], '[CH2;D2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH3]', "[CH2]CCCCCCCCC", 0.2),
    Substitution(['CO2CH2Bn'], '[C](=O)O[CH2]c1ccccc1', "[C](=O)O[CH2]c1ccccc1", 0.2),
    Substitution(['NMe'], '[N]C', "[N]C", 0.2),
    Substitution(['TIPS','TlPS'], '[Si](C(C)C)(C(C)C)C(C)C', '[Si](C(C)C)(C(C)C)C(C)C', 0.2),
    Substitution(['C6F5','C8F5'], ' c1c(F)c(F)c(F)c(F)c1(F)', ' [c]1c(F)c(F)c(F)c(F)c1(F)', 0.2),
    Substitution(['OC6Cl5'], '[O]c1c(Cl)c(Cl)c(Cl)c(Cl)c1(Cl)', '[O]c1c(Cl)c(Cl)c(Cl)c(Cl)c1(Cl)', 0.2),
    Substitution(['nC5H11','C5H11'], 'CCCCC', 'CCCCC', 0.2),
    Substitution(['nC4H9','C4H9'], 'CCCC', 'CCCC', 0.2),
    Substitution(['pTol','Tol'], '[c]1ccc(C)cc1', '[c]1ccc(C)cc1', 0.2),
    Substitution(['PMP'], '[C]2=CC=C(OC)C=C2', '[C]2=CC=C(OC)C=C2', 0.2),
    Substitution(['OPMP'], '[O]C2=CC=C(OC)C=C2', '[O]C2=CC=C(OC)C=C2', 0.2),
    Substitution(['NMe2','Me2N'], '[N](C)C', '[N](C)C', 0.2),
    Substitution(['C(O)NMe2'], '[C](=O)N(C)C', '[C](=O)N(C)C', 0.2),
    Substitution(['C(O)Et'], '[C](=O)CC', '[C](=O)CC', 0.2),
    Substitution(['CHPh2',"CH(Ph)2"], '[CH](c1ccccc1)c1ccccc1', '[CH](c1ccccc1)c1ccccc1', 0.2),



    Substitution(['4-BrC6H4','BrC6H4'], 'c1ccc(Br)cc1', "c1ccc(Br)cc1", 0.4),
    Substitution(['2-BrC6H4'], 'c1c(Br)cccc1', "c1c(Br)cccc1", 0.4),
    Substitution(['3-BrC6H4'], 'c1cc(Br)ccc1', "c1cc(Br)ccc1", 0.4),
    Substitution(['CF3C6H3','(CF3)C6H3'], '[C](F)(F)F', "[C](F)(F)F", 0.4),
    Substitution(['4-CO2MeC6H4'], 'C(=O)Oc1ccc(C)cc1', "[c]1ccc(C(=O)OC)cc1", 0.5),
    Substitution(['3-CO2MeC6H4'], 'C(=O)Oc1cc(C)ccc1', "[c]1cc(C(=O)OC)ccc1", 0.5),
    Substitution(['Napdh','17Napdh'], 'c1ccc2ccccc2c1', "c1ccc2ccccc2c1",0.5),
    Substitution(['2-MeC6H4'], 'c1c(C)cccc1', "c1c(C)cccc1", 0.5),
    Substitution(['3-MeC6H4'], 'c1cc(C)ccc1', "c1cc(C)ccc1", 0.5),
    Substitution(['4-MeC6H4','MeC6H4','AeC6H4','4MeC6H4'], 'c1ccc(C)cc1', "c1ccc(C)cc1", 0.5),
    Substitution(['4-OMeC6H4','OMeC6H4'], 'c1ccc(OC)cc1', "c1ccc(OC)cc1", 0.5),
    Substitution(['3-OMeC6H4','3OMeC6H4'], 'c1cc(OC)ccc1', "c1cc(OC)ccc1", 0.5),
    Substitution(['2-OMeC6H4'], 'c1c(OC)cccc1', "c1c(OC)cccc1", 0.5),
    Substitution(['4-MeOC6H4','MeOC6H4','p-MeO-C6H4','4-MeO'], 'c1ccc(OC)cc1', "c1ccc(OC)cc1", 0.5),
    Substitution(['3-MeOC6H4','3MeOC6H4'], 'c1cc(OC)ccc1', "c1cc(OC)ccc1", 0.5),
    Substitution(['2-MeOC6H4','C6H4OMe-2'], 'c1c(OC)cccc1', "c1c(OC)cccc1", 0.5),
    Substitution(['2-ClC6H4'], 'c1c(Cl)cccc1', "c1c(Cl)cccc1", 0.2),
    Substitution(['4-ClC6H4','C6H4Cl-4','ClC6H4'], 'c1ccc(Cl)cc1', "c1ccc(Cl)cc1", 0.5),
    Substitution(['4-FC6H4'], 'c1ccc(F)cc1', "c1ccc(F)cc1", 0.5),  
    Substitution(['4-CF3C6H4','A4CF3C6H4','4CF3C6H4'], 'c1ccc(cc1)C(F)(F)F', "c1ccc(cc1)C(F)(F)F", 0.5),
    Substitution(['4-NO2C6H4','NO2C6H4','PNP','4NO2C6H4'], '[c]1ccc(cc1)[N+](=O)[O-]', "[c]1ccc(cc1)[N+](=O)[O-]", 0.5),
    Substitution(['2-thienyl'], '[c]1[s]ccc1', "[c]1[s]ccc1", 0.5),
    Substitution(['2-furyl','Z'], '[c]1occc1', "[c]1occc1", 0.5),
    Substitution(['2-pyridyl'], 'c1ncccc1', "c1ncccc1", 0.5),
    Substitution(['3-pyridyl'], '[c]1cccnc1', "[c]1cccnc1", 0.5),  # was c1ccncc1 (4-pyridyl)
    Substitution(['2,4-Cl2C6H3','2, 4-Cl2C6H3','Cl2C6H3'], 'c1c(Cl)cc(Cl)cc1', "c1c(Cl)cc(Cl)cc1", 0.5),

    Substitution(['[CF3]2C6H3', '3,5-[CF3]2C6H3','3,5-(CF3)2C6H3', '3,5-CF3C6H3'], '[c]1cc(C(F)(F)F)cc(C(F)(F)F)c1', "[c]1cc(C(F)(F)F)cc(C(F)(F)F)c1", 0.4),
    Substitution(['B(OH)2','(HO)2B'], '[B]([OH])([OH])', "B(O)O", 0.4),
    Substitution(['NPhth', 'NPthh'], '[N]1C(=O)c2ccccc2C1=O', "[N]1C(=O)c2ccccc2C1=O", 0.5),  # was open-chain
    Substitution(['1-Nap'], '[c]1cccc2ccccc12', "[c]1cccc2ccccc12", 0.5),
    Substitution(['PhCH2'], '[CH2]c1ccccc1', "[CH2]c1ccccc1", 0.5),  # was [C]c1ccccc1
   ###NEW
    Substitution(['B(OH)2','B(0H)2'],'B(O)O','B(O)O',0.5),
    Substitution(['CF2H','HF2C','F2C'],'C(F)(F)','C(F)(F)',0.5),
    Substitution(['SCF3','ScF3'],'SC(F)(F)F','SC(F)(F)F',0.5),
    Substitution(['F3'],'(F)(F)F','(F)(F)F',0.5),
    Substitution(['AgSe','AgScF3'],'[Ag+].[S-]C(F)(F)F','[Ag+].[S-]C(F)(F)F',0.5),
    Substitution(['Me3Si'],'[Si](C)(C)C','[Si](C)(C)C',0.5),
    Substitution(['CN'],'C#N','C#N',0.5),
    Substitution(['SN'],'SN','SN',0.5),

    Substitution(['N((SO2Ph))2'],'N(S(=O)(=O)c1ccccc1)S(=O)(=O)c1ccccc1','N(S(=O)(=O)c1ccccc1)S(=O)(=O)c1ccccc1',0.5),
    Substitution(['NHTs'],'NS(=O)(=O)c1ccc(cc1)C','NS(=O)(=O)c1ccc(cc1)C',0.5),
    Substitution(['OCOCH3'],'OC(=O)C','OC(=O)C',0.5),
    Substitution(['CO2tBu', 'CO2tBu'], 'C(=O)OC(C)(C)C', "C(=O)OC(C)(C)C", 0.3),# this substitution seems a bit off
    Substitution(['ScH3','SCH3'],'SC','SC',0.5),
    Substitution(['n-Pent'],'CCCCC','CCCCC',0.5),

    Substitution(['Si(OEt)3'], '[Si](OCC)(OCC)OCC', '[Si](OCC)(OCC)OCC', 0.2),
    Substitution(['SiPhMe2'], '[Si](C)(C)c1ccccc1', '[Si](C)(C)c1ccccc1', 0.2),  # was unparsable

    ### 2026-09-14: spellings found by surveying every symbol of the 324-image benchmark runs
    ### (analysis_gap/x_symbol_survey.py); each entry checked by expanding it on a graph and against the GT
    # standalone species (text boxes read as one "molecule")
    Substitution(['ClO4-', 'ClO4'], "Cl([O-])(=O)(=O)=O", "Cl([O-])(=O)(=O)=O", 0),
    Substitution(['A-HBF4', '20BF4', '38F4', 'HBF'], 'F[B-](F)(F)F', "F[B-](F)(F)F", 0),  # OCR of BF4 / HBF4
    Substitution(['AgSCF3'], '[Ag+].[S-]C(F)(F)F', '[Ag+].[S-]C(F)(F)F', 0),
    Substitution(['KOtBu', 't-BuOK', 'tBuOK'], 'CC(C)(C)[O-].[K+]', 'CC(C)(C)[O-].[K+]', 0),
    Substitution(['NiCl2'], 'Cl[Ni]Cl', 'Cl[Ni]Cl', 0),
    Substitution(['CS2'], 'S=C=S', 'S=C=S', 0),
    Substitution(['CH3SSO3Na'], 'CSS(=O)(=O)[O-].[Na+]', 'CSS(=O)(=O)[O-].[Na+]', 0),
    Substitution(['Ph3P+CF2CO2-', 'Ph3PCF2CO2'], '[O-]C(=O)C(F)(F)[P+](c1ccccc1)(c1ccccc1)c1ccccc1', '[O-]C(=O)C(F)(F)[P+](c1ccccc1)(c1ccccc1)c1ccccc1', 0),
    Substitution(['2-CF3C6H4OH'], 'Oc1ccccc1C(F)(F)F', 'Oc1ccccc1C(F)(F)F', 0),
    Substitution(['2-MeOC6H4OH'], 'Oc1ccccc1OC', 'Oc1ccccc1OC', 0),
    Substitution(['2-PhC6H4OH'], 'Oc1ccccc1-c1ccccc1', 'Oc1ccccc1-c1ccccc1', 0),
    Substitution(['2-i-PrOC6H4OH'], 'Oc1ccccc1OC(C)C', 'Oc1ccccc1OC(C)C', 0),
    Substitution(['2-i-PrC6H4OH'], 'Oc1ccccc1C(C)C', 'Oc1ccccc1C(C)C', 0),
    Substitution(['2-t-BuC6H4OH'], 'Oc1ccccc1C(C)(C)C', 'Oc1ccccc1C(C)(C)C', 0),
    Substitution(['PhOH'], 'Oc1ccccc1', 'Oc1ccccc1', 0),
    # substituted phenyl: hyphenated, o/m/p and English-name spellings, OCR variants
    Substitution(['4-bromophenyl', '4-Br-C6H4', 'p-BrC6H4', '4-BrC8H4', 'LrC8H4'], '[c]1ccc(Br)cc1', '[c]1ccc(Br)cc1', 0.3),
    Substitution(['3-bromophenyl', '3BrC6H4'], '[c]1cccc(Br)c1', '[c]1cccc(Br)c1', 0.3),
    Substitution(['4-chlorophenyl', '4-Cl-C6H4', 'p-C6H4Cl', 'p-ClC6H4', 'pClC6H4', 'PC6H4Cl'], '[c]1ccc(Cl)cc1', '[c]1ccc(Cl)cc1', 0.3),
    Substitution(['3-chlorophenyl', '3-Cl-C6H4', '3-ClC6H4', 'm-ClC6H4'], '[c]1cccc(Cl)c1', '[c]1cccc(Cl)c1', 0.3),
    Substitution(['2-chlorophenyl', '2-Cl-C6H4', 'o-ClC6H4'], '[c]1ccccc1Cl', '[c]1ccccc1Cl', 0.3),
    Substitution(['4-fluorophenyl', '4-F-C6H4', 'p-FC6H4', 'A-FC6H4'], '[c]1ccc(F)cc1', '[c]1ccc(F)cc1', 0.3),
    Substitution(['3-fluorophenyl', '3-F-C6H4', '3-FC6H4'], '[c]1cccc(F)c1', '[c]1cccc(F)c1', 0.3),
    Substitution(['2-fluorophenyl', '2-F-C6H4', '2-FC6H4', 'o-FC6H4'], '[c]1ccccc1F', '[c]1ccccc1F', 0.3),
    Substitution(['4-methoxyphenyl', '4-MeO-C6H4', 'p-MeOC6H4', 'p-anisyl'], '[c]1ccc(OC)cc1', '[c]1ccc(OC)cc1', 0.3),
    Substitution(['2-methoxyphenyl', '2-MeO-C6H4', 'o-MeOC6H4'], '[c]1ccccc1OC', '[c]1ccccc1OC', 0.3),
    Substitution(['4-nitrophenyl', '4-NO2-C6H4', 'p-NO2C6H4'], '[c]1ccc([N+](=O)[O-])cc1', '[c]1ccc([N+](=O)[O-])cc1', 0.3),
    Substitution(['3-nitrophenyl', '3-NO2-C6H4', '3-NO2C6H4'], '[c]1cccc([N+](=O)[O-])c1', '[c]1cccc([N+](=O)[O-])c1', 0.3),
    Substitution(['2-nitrophenyl', '2-NO2-C6H4', '2-NO2C6H4'], '[c]1ccccc1[N+](=O)[O-]', '[c]1ccccc1[N+](=O)[O-]', 0.3),
    Substitution(['4-tolyl', 'p-tolyl', '4-Tol', 'p-Tol', 'P-Tol', 'PTol', 'PoTol', 'L4MeC6H4'], '[c]1ccc(C)cc1', '[c]1ccc(C)cc1', 0.3),
    Substitution(['3-methylphenyl', '3MeC6H4', 'm-MeC6H4', 'm-Tol'], '[c]1cccc(C)c1', '[c]1cccc(C)c1', 0.3),
    Substitution(['2-methylphenyl', 'o-MeC6H4', 'o-Tol'], '[c]1ccccc1C', '[c]1ccccc1C', 0.3),
    Substitution(['4-trifluoromethylphenyl', '4-CF3-C6H4', 'p-CF3C6H4'], '[c]1ccc(C(F)(F)F)cc1', '[c]1ccc(C(F)(F)F)cc1', 0.3),
    Substitution(['3-CF3-C6H4', '3-CF3C6H4'], '[c]1cccc(C(F)(F)F)c1', '[c]1cccc(C(F)(F)F)c1', 0.3),
    Substitution(['2-CF3-C6H4', '2-CF3C6H4'], '[c]1ccccc1C(F)(F)F', '[c]1ccccc1C(F)(F)F', 0.3),
    Substitution(['4-cyanophenyl', '4-CN-C6H4', '4-CNC6H4', '4-NCC6H4'], '[c]1ccc(C#N)cc1', '[c]1ccc(C#N)cc1', 0.3),
    Substitution(['4-isopropylphenyl', '4-i-PrC6H4', '4-iPrC6H4'], '[c]1ccc(C(C)C)cc1', '[c]1ccc(C(C)C)cc1', 0.3),
    Substitution(['2-i-PrC6H4', '2-iPrC6H4'], '[c]1ccccc1C(C)C', '[c]1ccccc1C(C)C', 0.3),
    Substitution(['4-tert-butylphenyl', '4-t-BuC6H4', '4-tBuC6H4'], '[c]1ccc(C(C)(C)C)cc1', '[c]1ccc(C(C)(C)C)cc1', 0.3),
    Substitution(['2-t-BuC6H4', '2-tBuC6H4'], '[c]1ccccc1C(C)(C)C', '[c]1ccccc1C(C)(C)C', 0.3),
    Substitution(['2-i-PrOC6H4', '2-iPrOC6H4'], '[c]1ccccc1OC(C)C', '[c]1ccccc1OC(C)C', 0.3),
    Substitution(['2-OHC6H4', '2-HOC6H4'], '[c]1ccccc1O', '[c]1ccccc1O', 0.3),
    Substitution(['4-carbomethoxyphenyl', '4-MeO2CC6H4'], '[c]1ccc(C(=O)OC)cc1', '[c]1ccc(C(=O)OC)cc1', 0.3),
    Substitution(['4-CO2EtC6H4', '4-EtO2CC6H4'], '[c]1ccc(C(=O)OCC)cc1', '[c]1ccc(C(=O)OCC)cc1', 0.3),
    Substitution(['4-COCH3C6H4', '4-MeCOC6H4', '4-AcC6H4'], '[c]1ccc(C(C)=O)cc1', '[c]1ccc(C(C)=O)cc1', 0.3),
    Substitution(['4-Me2NC6H4', '4-NMe2C6H4'], '[c]1ccc(N(C)C)cc1', '[c]1ccc(N(C)C)cc1', 0.3),
    Substitution(['2-biphenyl', '2-PhC6H4'], '[c]1ccccc1-c1ccccc1', '[c]1ccccc1-c1ccccc1', 0.3),
    Substitution(['4-biphenyl', '4-PhC6H4'], '[c]1ccc(-c2ccccc2)cc1', '[c]1ccc(-c2ccccc2)cc1', 0.3),
    Substitution(['2,6-dimethylC6H3', '2,6-Me2C6H3'], '[c]1c(C)cccc1C', '[c]1c(C)cccc1C', 0.3),
    Substitution(['2,6-Et2C6H3'], '[c]1c(CC)cccc1CC', '[c]1c(CC)cccc1CC', 0.3),
    Substitution(['3,4-dichlorophenyl', '3,4-Cl2C6H3'], '[c]1ccc(Cl)c(Cl)c1', '[c]1ccc(Cl)c(Cl)c1', 0.3),
    Substitution(['3,4-methylenedioxyphenyl', 'benzo-dioxole', 'benzodioxole'], '[c]1ccc2OCOc2c1', '[c]1ccc2OCOc2c1', 0.3),
    Substitution(['phenyl', 'C6H5'], '[c]1ccccc1', '[c]1ccccc1', 0.3),
    Substitution(['1-naphthyl', 'alpha-naphthyl', 'α-naphthyl', 'Napthyl-1', 'Napthyl', '1-Napth', 'Napth'], '[c]1cccc2ccccc12', '[c]1cccc2ccccc12', 0.3),  # GT of 146.jpg / 332.jpg: 1-naphthyl
    Substitution(['2-naphthyl', 'beta-naphthyl', 'β-naphthyl'], '[c]1ccc2ccccc2c1', '[c]1ccc2ccccc2c1', 0.3),
    Substitution(['C8F6'], '[c]1c(F)c(F)c(F)c(F)c1F', '[c]1c(F)c(F)c(F)c(F)c1F', 0.3),  # OCR of C6F5
    Substitution(['OC8Cl5'], '[O]c1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl', '[O]c1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl', 0.3),  # OCR of OC6Cl5
    # heteroaryl
    Substitution(['2-Thienyl'], '[c]1cccs1', '[c]1cccs1', 0.3),
    Substitution(['3-thienyl', '3-Thienyl'], '[c]1ccsc1', '[c]1ccsc1', 0.3),
    Substitution(['2-Furyl', 'Turyl'], '[c]1ccco1', '[c]1ccco1', 0.3),
    Substitution(['2-(5-(4-ClC6H4)-Furyl)'], '[c]1ccc(-c2ccc(Cl)cc2)o1', '[c]1ccc(-c2ccc(Cl)cc2)o1', 0.3),
    Substitution(['2-(4,5-Di-Me-Furyl)'], '[c]1cc(C)c(C)o1', '[c]1cc(C)c(C)o1', 0.3),
    Substitution(['2-(5-Me-Furyl)', '5-Me-2-furyl'], '[c]1ccc(C)o1', '[c]1ccc(C)o1', 0.3),
    Substitution(['2-Benzofuryl', '2-benzofuryl'], '[c]1cc2ccccc2o1', '[c]1cc2ccccc2o1', 0.3),
    Substitution(['2-Pyridyl'], '[c]1ccccn1', '[c]1ccccn1', 0.3),
    Substitution(['4-pyridyl', '4-Pyridyl'], '[c]1ccncc1', '[c]1ccncc1', 0.3),
    Substitution(['2-pyrrolyl', '2-Pyrryl', '2-pyrryl'], '[c]1ccc[nH]1', '[c]1ccc[nH]1', 0.3),
    # alkyl, alkenyl, aralkyl
    Substitution(['n-propyl', 'propyl', 'n-C3H7'], '[CH2]CC', '[CH2]CC', 0.3),  # n-propyl went to a name service and came back as propylamine
    Substitution(['n-butyl', 'butyl', '1-Butyl'], '[CH2]CCC', '[CH2]CCC', 0.3),
    Substitution(['n-pentyl', 'pentyl', 'n-C5H11'], '[CH2]CCCC', '[CH2]CCCC', 0.3),
    Substitution(['methyl', 'Ie'], '[CH3]', '[CH3]', 0.3),  # Ie: OCR of Me (152_image_3_1)
    Substitution(['ethyl'], '[CH2]C', '[CH2]C', 0.3),
    Substitution(['tert-butyl', 't-butyl', 'tbutyl'], '[C](C)(C)C', '[C](C)(C)C', 0.3),
    Substitution(['cyclohexyl', '1-c-C6H11', 'c-C6H11', 'c-Hex'], '[CH]1CCCCC1', '[CH]1CCCCC1', 0.3),
    Substitution(['p-methoxybenzyl'], '[CH2]c1ccc(OC)cc1', '[CH2]c1ccc(OC)cc1', 0.3),
    Substitution(['benzyl'], '[CH2]c1ccccc1', '[CH2]c1ccccc1', 0.3),
    Substitution(['Ph2CH'], '[CH](c1ccccc1)c1ccccc1', '[CH](c1ccccc1)c1ccccc1', 0.3),
    Substitution(['phenethyl', 'Ph(CH2)2', 'PhCH2CH2'], '[CH2]Cc1ccccc1', '[CH2]Cc1ccccc1', 0.3),
    Substitution(['(E)-CH=CHPh', 'CH=CHPh', 'styryl'], '[CH]=Cc1ccccc1', '[CH]=Cc1ccccc1', 0.3),
    Substitution(['1-propenyl', 'Me-CH=CH', 'MeCH=CH'], '[CH]=CC', '[CH]=CC', 0.3),
    Substitution(['CD3', 'D3C'], '[C]([2H])([2H])[2H]', '[C]([2H])([2H])[2H]', 0.3),
    Substitution(['CH2CH2(1,3-dithian-2-yl)', 'CH2CH2C1SCCCS1'], '[CH2]CC1SCCCS1', '[CH2]CC1SCCCS1', 0.3),
    # heteroatom-linked groups
    Substitution(['Me2NCO', 'CONMe2'], '[C](=O)N(C)C', '[C](=O)N(C)C', 0.3),
    Substitution(['CORMe'], '[C](=O)C', '[C](=O)C', 0.3),
    Substitution(['COREt'], '[C](=O)CC', '[C](=O)CC', 0.3),
    Substitution(['CORPh'], '[C](=O)c1ccccc1', '[C](=O)c1ccccc1', 0.3),
    Substitution(['NHCOAr'], '[NH]C(=O)*', '[NH]C(=O)*', 0.3),
    Substitution(['COAr'], '[C](=O)*', '[C](=O)*', 0.3),
    Substitution(['O-t-Bu', 'Ot-Bu', 't-BuO'], '[O]C(C)(C)C', '[O]C(C)(C)C', 0.3),
    Substitution(['PvO'], '[O]C(=O)C(C)(C)C', '[O]C(=O)C(C)(C)C', 0.3),  # OCR of PivO
    Substitution(['OTBi'], '[O][Si](C)(C)C(C)(C)C', '[O][Si](C)(C)C(C)(C)C', 0.3),  # OCR of OTBS
    Substitution(['OPO((OMe))2'], '[O]P(=O)(OC)OC', '[O]P(=O)(OC)OC', 0.3),
    Substitution(['BocNH', 'BocH'], '[NH]C(=O)OC(C)(C)C', '[NH]C(=O)OC(C)(C)C', 0.3),
    Substitution(['CbzH'], '[NH]C(=O)OCc1ccccc1', '[NH]C(=O)OCc1ccccc1', 0.3),  # OCR of CbzHN
    Substitution(['TsNH'], '[NH]S(=O)(=O)c1ccc(C)cc1', '[NH]S(=O)(=O)c1ccc(C)cc1', 0.3),
    Substitution(['TfaHN', 'NHTfa', 'TraHN'], '[NH]C(=O)C(F)(F)F', '[NH]C(=O)C(F)(F)F', 0.3),
    Substitution(['TrHNOC'], '[C](=O)NC(c1ccccc1)(c1ccccc1)c1ccccc1', '[C](=O)NC(c1ccccc1)(c1ccccc1)c1ccccc1', 0.3),
    Substitution(['PGHN'], '[NH]*', '[NH]*', 0.3),  # PG = any protecting group
    Substitution(['(CH2)4N', 'N(CH2)4'], '[N]1CCCC1', '[N]1CCCC1', 0.3),
    Substitution(['H2NO2S'], '[S](=O)(=O)N', '[S](=O)(=O)N', 0.3),
    Substitution(['SO3F'], '[S](=O)(=O)F', '[S](=O)(=O)F', 0.3),
    Substitution(['B((PpIn))', 'B(PpIn)', 'B(Ppin)', 'B(pIn)', 'Bpm', 'Bpn'], 'B1OC(C)(C)C(C)(C)O1', "B1OC(C)(C)C(C)(C)O1", 0.3),  # OCR of Bpin
    Substitution(['BF3K', 'KF3B'], '[B-](F)(F)F.[K+]', '[B-](F)(F)F.[K+]', 0.3),
    Substitution(['ZrCp2Cl', 'ZrGp2Cl'], '[Zr](Cl)(C1=CC=CC1)C1=CC=CC1', '[Zr](Cl)(C1=CC=CC1)C1=CC=CC1', 0.3),
    # Counter-ions written with their charge sign, as the vision model reads the label beside a salt
    # ("BF4-" drawn next to an azolium): the free anion. [BF4-] was read 141 times in the 2026-09-14
    # runs and, missing here, expanded to *. The mol-edit-plan charge rule also turns [BF4] into [BF4-].
    Substitution(['BF4-'], 'F[B-](F)(F)F', 'F[B-](F)(F)F', 0),
    Substitution(['PF6-', 'PF6'], 'F[P-](F)(F)(F)(F)F', 'F[P-](F)(F)(F)(F)F', 0),
    Substitution(['SbF6-', 'SbF6'], 'F[Sb-](F)(F)(F)(F)F', 'F[Sb-](F)(F)(F)(F)F', 0),
    Substitution(['OTf-', 'TfO-'], 'O=S(=O)([O-])C(F)(F)F', 'O=S(=O)([O-])C(F)(F)F', 0),
    Substitution(['NTf2-', 'Tf2N-'], 'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F', 'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F', 0),
]

ABBREVIATIONS = {abbrv: sub for sub in SUBSTITUTIONS for abbrv in sub.abbrvs}

VALENCES = {
    "H": [1], "Li": [1], "Be": [2], "B": [3], "C": [4], "N": [3, 5], "O": [2], "F": [1],
    "Na": [1], "Mg": [2], "Al": [3], "Si": [4], "P": [5, 3], "S": [6, 2, 4], "Cl": [1], "K": [1], "Ca": [2],
    "Br": [1], "I": [1]
}

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

COLORS = {
    u'c': '0.0,0.75,0.75', u'b': '0.0,0.0,1.0', u'g': '0.0,0.5,0.0', u'y': '0.75,0.75,0',
    u'k': '0.0,0.0,0.0', u'r': '1.0,0.0,0.0', u'm': '0.75,0,0.75'
}

# tokens of condensed formula
FORMULA_REGEX = re.compile(
    '(?:' + '|'.join([re.escape(k) for k in ABBREVIATIONS.keys()]) + '|R[0-9]*|[A-Z][a-z]+|[A-Z]|[0-9]+|\(|\))')
