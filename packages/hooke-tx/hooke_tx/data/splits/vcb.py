import json
import numpy as np

from hooke_tx.data.constants import CELL_TYPE, MOL_PERT_NAME


def custom_split_vcb(split_type, adatas, data_axis_indices):
    """
    Custom split for TREK_VCDS1 dataset.
    """
    mode = split_type.split("_")[-1]
    if mode in ["id"]:
        split_type = split_type.replace(f"_{mode}", "")

    if "drugscreen" in split_type:
        empty_pert_indices = [int(line.strip()) for line in open("/rxrx/scratch/hooke_tx/trek_vcb/drugscreen__trekseq__v1_1/empty_pert_indices.txt")]

        split_path = f"/rxrx/data/valence/internal_benchmarking/vcb/splits/drugscreen__trekseq__v1_1/split_compound_random__v1.json"

        if mode == "id":
            split_path = f"/rxrx/data/valence/internal_benchmarking/vcb/splits/drugscreen__trekseq__v1_1/split_observation_random__v1.json"

    elif "cross_cell_line" in split_type:
        empty_pert_indices = []
        split_path = f"/rxrx/data/valence/internal_benchmarking/vcb/splits/cross_cell_line_trek__v1_1/split_cell_type_manual__v1.json"
    
    else:
        raise ValueError(f"Split type {split_type} not supported")

    split = json.load(open(split_path))

    train_indices = split["folds"][0]["finetune"]
    val_indices = split["folds"][0]["validation"]
    test_indices = split["folds"][0]["test"]

    # Remove empty pert indices from train, val, and test indices
    train_indices = [idx for idx in train_indices if idx not in empty_pert_indices]
    val_indices = [idx for idx in val_indices if idx not in empty_pert_indices]
    test_indices = [idx for idx in test_indices if idx not in empty_pert_indices]
    
    split_indices = {}

    split_indices["train"] = {split_type: train_indices}
    split_indices["val"] = {}
    split_indices["val"]["default"] = {split_type: val_indices}
    split_indices["val"]["test"] = {split_type: test_indices}
    split_indices["test"] = {split_type: test_indices}
    
    # Iterate over data sources
    for src in data_axis_indices[CELL_TYPE].keys():
        if src != split_type:
            split_indices["train"][src] = []

            for c_indices in data_axis_indices[CELL_TYPE][src].values():
                c_indices = set(c_indices)

                for p, p_indices in data_axis_indices[MOL_PERT_NAME][src].items():
                    p_indices = set(p_indices)
                    shared_indices = list(c_indices & p_indices)
                    split_indices["train"][src].extend(shared_indices)

            split_indices["train"][src] = list(set(split_indices["train"][src]))
    
    # Define split_categories
    split_categories = {
        "train": [],
        "val": [],
        "test": []
    }

    # Make incdices unique
    for src in split_indices["train"].keys():
        if src != split_type:
            split_indices["train"][src] = list(set(split_indices["train"][src]))
    
    return split_indices, split_categories


def custom_split_vcb_cx(adatas, data_axis_indices, seed: int = 42):
    """
    Custom split for TREK_VCDS1 dataset.
    """
    assert "TREK_VCB_cross_cell_line_v1_2_fold_0" in adatas.keys(), "TREK_VCB_cross_cell_line_v1_2_fold_0 data source is required for custom_split_vcb_cx split"
    
    split_indices = {}
    split_indices["train"] = {src: [] for src in adatas.keys()}
    split_indices["val"] = {}
    split_indices["val"]["default"] = {"TREK_VCB_cross_cell_line_v1_2_fold_0": []}
    split_indices["test"] = {"TREK_VCB_cross_cell_line_v1_2_fold_0": []}

    val_cell_types = ["NCI-H1792", "NCI-H23"]
    val_fraction = 1/2

    test_cell_types = ["ES2", "OVCAR3", "NCI-H3122", "NCI-H1703"]
    test_fraction = 2/3

    val_test_compounds = {
        "NCI-H1792": [
            'Cc1ccc2nc(C)c(C(=O)N3CCN(c4ccc(S(C)(=O)=O)cc4[N+](=O)[O-])CC3)cc2c1',
            'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@H](CC(=O)OC(C)(C)C)c1nnc(C)n1-2',
            'O=C1CC2(CCN(C(=O)NCc3ccc(F)cc3)CC2)Nc2ccc(F)cc21',
            'Cc1ccc(OCCNc2ccc(S(=O)(=O)N3CCN(C)CC3)cc2[N+](=O)[O-])c(C)c1'
        ],
        "NCI-H23": [
            'Cc1ccc2nc(C)c(C(=O)N3CCN(c4ccc(S(C)(=O)=O)cc4[N+](=O)[O-])CC3)cc2c1',
            'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@H](CC(=O)OC(C)(C)C)c1nnc(C)n1-2',
            'O=C1CC2(CCN(C(=O)NCc3ccc(F)cc3)CC2)Nc2ccc(F)cc21',
            'Cc1ccc(OCCNc2ccc(S(=O)(=O)N3CCN(C)CC3)cc2[N+](=O)[O-])c(C)c1'
        ],
        "ES2": [
            'COCc1cc(C(=O)Nc2cc([C@H]3CC[C@@H](OC(=O)NC(C)C)C3)n[nH]2)n(C)n1',
            'O=c1[nH]ncc(N2CCC(Oc3ccc([N+](=O)[O-])cc3)CC2)c1Cl',
            'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@H](CC(=O)OC(C)(C)C)c1nnc(C)n1-2',
            'Cc1nnc2ccc(N3CCN(C(=O)COc4ccc(Cl)cc4)CC3)nn12',
            'CN(C)C/C=C/C(=O)Nc1ccc(C(=O)Nc2ccc(OCc3cn4ccccc4n3)cc2)cc1',
            'C[C@@]1(O)CCC[C@H]1n1c(=O)c(C(F)F)cc2cnc(NC3CCN(S(C)(=O)=O)CC3)nc21'
        ],
        "OVCAR3": [
            'COCc1cc(C(=O)Nc2cc([C@H]3CC[C@@H](OC(=O)NC(C)C)C3)n[nH]2)n(C)n1',
            'O=c1[nH]ncc(N2CCC(Oc3ccc([N+](=O)[O-])cc3)CC2)c1Cl',
            'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@H](CC(=O)OC(C)(C)C)c1nnc(C)n1-2',
            'Cc1nnc2ccc(N3CCN(C(=O)COc4ccc(Cl)cc4)CC3)nn12',
            'CN(C)C/C=C/C(=O)Nc1ccc(C(=O)Nc2ccc(OCc3cn4ccccc4n3)cc2)cc1',
            'C[C@@]1(O)CCC[C@H]1n1c(=O)c(C(F)F)cc2cnc(NC3CCN(S(C)(=O)=O)CC3)nc21'
        ],
        "NCI-H3122": [
            'O=C(Cc1sc(-c2ccccc2)nc1-c1ccccc1)Nc1ccc(Cl)cc1',
            'Cc1ccc(-c2nc(-c3cccnc3)sc2CC(=O)Nc2cccc(S(=O)(=O)N(C)C)c2)cc1',
            'O[C@@](Cn1cnnn1)(c1ccc(F)cc1F)C(F)(F)c1ccc(-c2ccc(OC(F)(F)F)cc2)cn1',
            'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@H](CC(=O)OC(C)(C)C)c1nnc(C)n1-2',
            'Cc1nc(-c2cn3c(n2)-c2ccc(-c4cnn(C(C)(C)C(N)=O)c4)cc2OCC3)n(C(C)C)n1',
            'Cc1ccc(Nc2cc(N(C)C)nc(N3CCN(c4cc(C)no4)CC3)n2)cc1', 'O=C1/C(=N\\c2cccc(C(F)(F)F)c2)c2ccccc2N1c1ccccc1',
            'CC[C@@H](C)Nc1cc(C(=O)N[C@H]2C[C@H]3CC[C@@H](C2)N3c2ccc(C(=O)C3CC3)cn2)c(C)cc1C(N)=O',
            'CCN(C)c1cc(Nc2ccc(C)cc2)nc(N2CCN(c3ccccc3F)CC2)n1',
            'Cc1ccc(Nc2cc(N(C)C)nc(N3CCN(c4ccccc4F)CC3)n2)cc1',
        ],
        "NCI-H1703": [
            'C[C@@H]1C[C@H](C)CN(S(=O)(=O)c2ccc3c(c2)C(=NO)c2cc(S(=O)(=O)N4C[C@H](C)C[C@H](C)C4)ccc2C3=NO)C1',
            'O=C(Nc1ccc2c(c1)OCCCO2)c1ccc(NC(=O)C2CC2)s1', 'Cc1sc2c(c1C)C(c1ccc(Cl)cc1)=N[C@H](CC(=O)OC(C)(C)C)c1nnc(C)n1-2',
            'CCC(=O)Nc1cc(NC(=O)[C@H]2Cc3cc(Cl)ccc3O2)ccc1F',
            'Cc1ccc(Oc2nc3ccccn3c(=O)c2C=C(C#N)C(=O)NC2CCS(=O)(=O)C2)cc1',
            'C=CC(=O)Nc1cc(NC(=O)c2ccc(OC(C)(C)C)cc2)ccc1F',
            'CC(C)(C)N1CC(C(=O)Nc2ccc(C(=O)Nc3ccc4c(c3)OCCCO4)s2)C1',
            'C=CC(=O)Nc1ccc(C(=O)Nc2ccc3c(c2)OCCCO3)s1',
            'Nc1ccc(F)cc1NC(=O)c1ccc(CNC(=O)/C=C/c2cccnc2)cc1',
            'Cc1cccn2c(=O)c(/C=C(\\C#N)C(=O)N(C)C34CC(C3)C4)c(Oc3ccc(F)cc3)nc12',
            'C=CC(=O)Nc1ccc2ncnc(Nc3ccc(OCc4cccc(F)c4)c(Cl)c3)c2c1',
            'C=CC(=O)Nc1cc(NC(=O)C2Cc3cc(Cl)ccc3O2)ccc1F',
            'Cc1cccn2c(=O)c(/C=C(\\C#N)C(=O)NC3CCCC3)c(Oc3ccc(F)cc3)nc12'
        ]
    }

    # Sample val/test compounds per cell type
    rng = np.random.RandomState(seed)
    selected_val_test_compounds = {}
    for cell_type in val_cell_types:
        compound_candidates = val_test_compounds[cell_type]
        n = int(val_fraction * len(compound_candidates))
        selected_compounds = rng.choice(compound_candidates, size=n, replace=False)
        selected_val_test_compounds[cell_type] = selected_compounds

    for cell_type in test_cell_types:
        compound_candidates = val_test_compounds[cell_type]
        n = int(test_fraction * len(compound_candidates))
        selected_compounds = rng.choice(compound_candidates, size=n, replace=False)
        selected_val_test_compounds[cell_type] = selected_compounds
    
    # Iterate over data sources
    for src in data_axis_indices[CELL_TYPE].keys():
        if src != "TREK_VCB_cross_cell_line_v1_2_fold_0":
            split_indices["train"][src] = []

            for c_indices in data_axis_indices[CELL_TYPE][src].values():
                c_indices = set(c_indices)

                for p, p_indices in data_axis_indices[MOL_PERT_NAME][src].items():
                    if p in ["NEG_CTRL", "UNKOWN"]:
                        continue

                    p_indices = set(p_indices)
                    shared_indices = list(c_indices & p_indices)
                    split_indices["train"][src].extend(shared_indices)

            split_indices["train"][src] = list(set(split_indices["train"][src]))
        
        else:
            for c, c_indices in data_axis_indices[CELL_TYPE][src].items():
                c_indices = set(c_indices)

                for p, p_indices in data_axis_indices[MOL_PERT_NAME][src].items():
                    if p in ["NEG_CTRL", "UNKOWN"]:
                        continue
                    
                    p_indices = set(p_indices)
                    shared_indices = list(c_indices & p_indices)
                    if c in val_cell_types and p in selected_val_test_compounds[c]:
                        split_indices["val"]["default"][src].extend(shared_indices)
                    elif c in test_cell_types and p in selected_val_test_compounds[c]:
                        split_indices["test"][src].extend(shared_indices)
                    else:
                        split_indices["train"][src].extend(shared_indices)

                if c in val_cell_types:
                    split_indices["val"]["default"][src].extend(c_indices)
                elif c in test_cell_types:
                    split_indices["test"][src].extend(c_indices)

    # Make indices unique
    for src in split_indices["train"].keys():
        if src != "TREK_VCB_drugscreen_v1_1_fold_0":
            split_indices["train"][src] = list(set(split_indices["train"][src]))
    
    # Define split_categories
    split_categories = {
        "train": [],
        "val": [],
        "test": []
    }

    split_indices["val"]["test"] = split_indices["test"]
    
    return split_indices, split_categories