import numpy as np
import toml

from hooke_tx.data.constants import CELL_TYPE, MOL_PERT_NAME, BASE_LABEL


def split_explicit_fewshot_state(adatas, data_axis_indices, data_axis_unique_values, fewshot_train_frac):
    split_indices = {
        "train": {src: [] for src in adatas.keys()},
        "val": {src: [] for src in adatas.keys()},
        "test": {src: [] for src in adatas.keys()},
    }

    all_contexts = list(data_axis_unique_values[CELL_TYPE])
    split_val = ["C32", "HOP62", "HepG2/C3A", "PANC-1", "Hs 766T"]
    split_test = split_val
    split_train = np.setdiff1d(all_contexts, split_val).tolist()

    all_conditions = [c for c in data_axis_unique_values[MOL_PERT_NAME] if c != BASE_LABEL]
    
    # randomly select conditions to hold out
    # target_conditions is the list of conditions to remove from train split for specified contexts
    unseen_frac = 1 - fewshot_train_frac
    iid_flag = False

    # custom split to match STATE fewshot evaluation
    # split source: https://huggingface.co/datasets/arcinstitute/State-Tahoe-Filtered/blob/main/generalization.toml
    # note that held out conditions are the same for each cell line
    state_split = toml.load("/rxrx/data/user/liam.hodgson/outgoing/tahoe/generalization.toml")
    val_conditions = [str(eval(p)[0]) for p in state_split["fewshot"]["tahoe_holdout.C32"]["val"]]
    test_conditions = [str(eval(p)[0]) for p in state_split["fewshot"]["tahoe_holdout.C32"]["test"]]
    val_test_conditions = val_conditions + test_conditions

    # iterate over data sources
    for src, context_lookup in data_axis_indices[CELL_TYPE].items():
        # iterate over contexts and get corresponding indices
        for target_context, context_indices in context_lookup.items():
            context_indices = set(context_indices)

            # get indices that match target conditions
            for target_condition in all_conditions:
                try:
                    condition_indices = set(data_axis_indices[MOL_PERT_NAME][src][target_condition])
                except:
                    continue

                # assign split based on context and condition
                # TODO: iid eval assigns same indices to multiple splits (temporary for debugging)
                assigned_splits = []
                if (target_context in split_train) or (target_condition not in val_test_conditions) or iid_flag:
                    # all conditions in train contexts or few-shot conditions in val/test context
                    assigned_splits.append("train")
                if (target_context in split_val) and (target_condition in val_conditions):
                    # held-out conditions in val context
                    assigned_splits.append("val")
                if (target_context in split_test) and (target_condition in test_conditions):
                    # held-out conditions in test context
                    assigned_splits.append("test")

                # save intersection of indices
                shared_indices = list(context_indices & condition_indices)
                for split_name in assigned_splits:
                    split_indices[split_name][src].extend(shared_indices)
    
    return split_indices