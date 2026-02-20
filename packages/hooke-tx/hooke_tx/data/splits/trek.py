import json
import numpy as np

from hooke_tx.data.constants import GENE_PERT, MOL_PERT_NAME, DOSE, BASE, BATCH


def custom_split_trek(split_type, adatas, data_axis_indices):
    """
    Custom split for TREK dataset.
    """
    split_indices = {}

    split_indices["train"] = {src: [] for src in adatas.keys()}
    split_indices["val"] = {}
    split_indices["test"] = {}

    split_indices["val"]["id_undiseased_seen"] = {src: [] for src in adatas.keys()}
    split_indices["val"]["id_undiseased_unseen"] = {src: [] for src in adatas.keys()}
    split_indices["val"]["id_diseased_seen"] = {src: [] for src in adatas.keys()}
    split_indices["val"]["id_diseased_unseen"] = {src: [] for src in adatas.keys()}
    split_indices["val"]["fewshot_from_undiseased"] = {src: [] for src in adatas.keys()}
    split_indices["val"]["fewshot_from_diseased"] = {src: [] for src in adatas.keys()}
    split_indices["val"]["ood_compound"] = {src: [] for src in adatas.keys()}

    context_lookup = {
        "id_undiseased": ["EMPTY"],
        "id_diseased": ["MBNL1"],
        "fewshot_from_undiseased": ["TSC2"],
        "fewshot_from_diseased": ["STK11"],
        "ood_compound": ["MBNL1"],
    }

    compound_lookup = {
        "id_undiseased": [
            "CC(C)n1c(/C=C/C(O)CC(O)CC(=O)[O-])c(-c2ccc(F)cc2)c2ccccc21",
            "Cn1cc(CNCC2CCN(c3ncc(C(=O)NO)cn3)CC2)c2ccccc21",
            "CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3c(CN(C)C)c(O)ccc3nc2-1"
        ],
        "id_diseased": [
            "CCN(C)S(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4cnc(C5CC5)nc4)cc23)c1F",
            "Cc1cc(-c2ccc(C3(O)CCC3)cc2)cc2cc[nH]c12",
            "COCc1ccc(-c2ccc3c(CC#N)c[nH]c3c2)cc1F"
        ],
        "fewshot_from_undiseased": [
            "CNC(=O)c1cccc(-c2ccc3c(N4C5CCC4COC5)nc(N4CCOC[C@@H]4C)nc3n2)c1",
            "C[C@@H]1COCCN1c1cc(C2(S(C)(=O)=O)CC2)nc(-c2cccc3[nH]ccc23)n1",
            "CNC(=O)c1cccc(-c2ccc3c(N4CCOC[C@@H]4C)nc(N4CCOC[C@@H]4C)nc3n2)c1"
        ],
        "fewshot_from_diseased": [
            "O=C(O)c1c[nH]c2cc(Cl)c(-c3ccc(C4(O)CCC4)cc3)cc12",
            "CCN(CCCCNC(=O)[C@@H]1C[C@H]1C(=O)NCCCCN(CC)S(=O)(=O)c1c(C)cc(C)cc1C)S(=O)(=O)c1c(C)cc(C)cc1C",
        ],
        "ood_compound": [
            "N#CCc1c[nH]c2cc(-c3ccc(CO)c(Cl)c3)ccc12",
            "O=c1sn(-c2cccc3ccccc23)c(=O)n1Cc1ccccc1",
            "OC1(c2ccc(-c3ccc4[nH]ccc4c3)cc2)CCC1"
        ],
    }

    dose_lookup = {
        "id_undiseased": [100.0, 1000.0, 10000.0],
        "id_diseased": [2000.0, 5000.0, 10000.0],
    }

    batch_lookup = {
        "id_undiseased_seen":[236, 237, 238, 239],
        "id_undiseased_unseen": [240, 241, 242, 243],
        "id_diseased_seen": [559, 566, 573, 580, 587],
        "id_diseased_unseen": [594, 601, 608, 615, 622]
    }
    
    # Define split_categories
    split_categories = {
        "train": [],
        "val": [],
        "test": []
    }

    relevant_contexts = []
    for context in context_lookup.values():
        relevant_contexts.extend(context)

    relevant_contexts = list(set(relevant_contexts))

    relevant_conditions = []
    for condition in compound_lookup.values():
        relevant_conditions.extend(condition)

    relevant_conditions = list(set(relevant_conditions))
    
    # Define diseased contexts (all contexts except EMPTY)
    diseased_contexts = [c for c in relevant_contexts if c != "EMPTY"]
    
    # Iterate over data sources
    for src, src_indices in data_axis_indices[GENE_PERT].items():
        control_indices = set(data_axis_indices[BASE][src])

        if src not in ["TREK_B_filtered_by_batch", "TREK_B_unfiltered_by_batch"]:
            split_indices["train"][src] = []

            for c_indices in data_axis_indices[GENE_PERT][src].values():
                c_indices = set(c_indices)

                for p, p_indices in data_axis_indices[MOL_PERT_NAME][src].items():
                    p_indices = set(p_indices)
                    shared_indices = list(c_indices & p_indices)
                    split_indices["train"][src].extend(shared_indices)

            split_indices["train"][src] = list(set(split_indices["train"][src]))
            continue
        
        # Determine batch prefix based on source name
        if "unfiltered" in src:
            batch_prefix = "TREK_B_unfiltered_"
        else:
            batch_prefix = "TREK_B_filtered_"
        
        # Iterate over contexts and get corresponding indices
        for context, context_indices in src_indices.items():
            context_indices = set(context_indices)

            if context not in relevant_contexts:
                # Exclude control indices from irrelevant contexts
                split_indices["train"][src].extend(list(context_indices - control_indices))
                continue

            if context in context_lookup["id_undiseased"]:
                for p in compound_lookup["id_undiseased"]:
                    p_indices = set(data_axis_indices[MOL_PERT_NAME][src][p])

                    for d in dose_lookup["id_undiseased"]:
                        d_indices = set(data_axis_indices[DOSE][src][d])
                        
                        # intersection of context_indices, p_indices and d_indices
                        shared_indices = list(context_indices & p_indices & d_indices)
                        
                        # partition based on batch membership
                        partition_seen = []
                        partition_unseen = []
                        partition_train = []
                        
                        for idx in shared_indices:
                            # Check batch membership to allocate to seen/unseen/train
                            if any(idx in data_axis_indices[BATCH][src].get(batch_prefix + str(b), []) for b in batch_lookup["id_undiseased_seen"]):
                                partition_seen.append(idx)
                                partition_train.append(idx)
                            elif any(idx in data_axis_indices[BATCH][src].get(batch_prefix + str(b), []) for b in batch_lookup["id_undiseased_unseen"]):
                                partition_unseen.append(idx)
                            else:
                                partition_train.append(idx)
                        
                        # save partition indices
                        split_indices["val"]["id_undiseased_unseen"][src].extend(partition_unseen)
                        split_indices["val"]["id_undiseased_seen"][src].extend(partition_seen)
                        split_indices["train"][src].extend(partition_train)

            if context in context_lookup["id_diseased"]:
                for p in compound_lookup["id_diseased"]:
                    p_indices = set(data_axis_indices[MOL_PERT_NAME][src][p])

                    for d in dose_lookup["id_diseased"]:
                        d_indices = set(data_axis_indices[DOSE][src][d])
                        
                        # intersection of context_indices, p_indices and d_indices
                        shared_indices = list(context_indices & p_indices & d_indices)

                        # partition based on batch membership
                        partition_seen = []
                        partition_unseen = []
                        partition_train = []
                        
                        for idx in shared_indices:
                            # Get batch number for this index
                            if any(idx in data_axis_indices[BATCH][src].get(batch_prefix + str(b), []) for b in batch_lookup["id_diseased_seen"]):
                                partition_seen.append(idx)
                                partition_train.append(idx)
                            elif any(idx in data_axis_indices[BATCH][src].get(batch_prefix + str(b), []) for b in batch_lookup["id_diseased_unseen"]):
                                partition_unseen.append(idx)
                            else:
                                partition_train.append(idx)
                        
                        # save partition indices
                        split_indices["val"]["id_diseased_unseen"][src].extend(partition_unseen)
                        split_indices["val"]["id_diseased_seen"][src].extend(partition_seen)
                        split_indices["train"][src].extend(partition_train)

            if context in context_lookup["fewshot_from_undiseased"]:
                for p in compound_lookup["fewshot_from_undiseased"]:
                    p_indices = set(data_axis_indices[MOL_PERT_NAME][src][p])
                    
                    shared_indices = list(context_indices & p_indices)

                    split_indices["val"]["fewshot_from_undiseased"][src].extend(shared_indices)

            if context in context_lookup["fewshot_from_diseased"]:
                for p in compound_lookup["fewshot_from_diseased"]:
                    p_indices = set(data_axis_indices[MOL_PERT_NAME][src][p])
                    
                    shared_indices = list(context_indices & p_indices)

                    split_indices["val"]["fewshot_from_diseased"][src].extend(shared_indices)

            if context in context_lookup["ood_compound"]:
                for p in compound_lookup["ood_compound"]:
                    p_indices = set(data_axis_indices[MOL_PERT_NAME][src][p])
                    
                    shared_indices = list(context_indices & p_indices)

                    split_indices["val"]["ood_compound"][src].extend(shared_indices)

            if context in relevant_contexts:
                for p in data_axis_indices[MOL_PERT_NAME][src].keys():
                    # Exclude ood_compound compounds entirely from training
                    if p in compound_lookup["ood_compound"]:
                        continue
                    
                    p_subsplits = [subsplit for subsplit, ps in compound_lookup.items() if p in ps]
                    
                    p_masked_contexts = []
                    for subsplit in p_subsplits:
                        if subsplit == "fewshot_from_undiseased":
                            # Mask ALL diseased contexts for fewshot_from_undiseased compounds
                            p_masked_contexts.extend(diseased_contexts)
                        else:
                            # For other subsplits, use the original context lookup
                            p_masked_contexts.extend(context_lookup[subsplit])
                    
                    p_masked_contexts = list(set(p_masked_contexts))

                    if context not in p_masked_contexts:
                        p_indices = set(data_axis_indices[MOL_PERT_NAME][src][p])
                        shared_indices = list(context_indices & p_indices)
                        split_indices["train"][src].extend(shared_indices)

    # Make incdices unique
    for src in split_indices["train"].keys():
        split_indices["train"][src] = list(set(split_indices["train"][src]))

    for subsplit in split_indices["val"].keys():
        for src in split_indices["val"][subsplit].keys():
            split_indices["val"][subsplit][src] = list(set(split_indices["val"][subsplit][src]))

    return split_indices, split_categories