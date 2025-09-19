import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from vcb.evaluate.utils import from_perturbations_to_compound_conc


class Baseline:
    """
    Base class for all baselines.
    """

    def __init__(
        self, train_dataset, valid_indices, reference_column: str = "is_base_state"
    ):
        """
        Args:
        train_dataset: Dataset object containing the training data.
        valid_indices: Indices of the training data to use for fitting the baseline (including controls).
        reference_column: Column to compare perturbed states to (e.g. "is_base_state" or "is_negative_control").
        """

        self.X = train_dataset.X
        self.valid_indices = np.array(valid_indices)

        self.feature_shape = train_dataset.X[0].shape

        self.obs = train_dataset.obs.with_row_index("X_index")
        self.obs = self.obs[self.valid_indices]

        # TODO: temporarily hardcoded
        self.CONTEXT_COL = "plate_disease_model"
        self.BATCH_COL = "batch_center"
        self.CONTROL_COL = reference_column
        self.PERT_COL = "perturbations"

        # dict: context -> batches -> aggregated array
        self.baseline_ctrls = defaultdict(
            lambda: defaultdict(partial(np.zeros, shape=self.feature_shape))
        )
        self.baseline_ctrls_count = defaultdict(lambda: defaultdict(lambda: 0))
        # dict: context -> perturbations -> batches -> aggregated array
        self.baseline_perts = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(partial(np.zeros, shape=self.feature_shape))
            )
        )
        self.baseline_perts_count = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: 0))
        )

        # group obs by context, perturbation, and batch
        self.obs = self.obs.group_by(
            self.CONTEXT_COL,
            self.PERT_COL,
            self.CONTROL_COL,
            self.BATCH_COL,
            maintain_order=True,
        ).agg(pl.col("X_index"))

        self._cache_group_means()

    def _cache_group_means(self):
        """
        Cache the mean of the observations by context, perturbation, and batch
        in preparation for baseline calculation.
        """
        # first, aggregate observations by context, perturbation, and batch
        for row in tqdm(
            self.obs.iter_rows(named=True),
            total=len(self.obs),
            desc="Aggregating by context, perturbation, and batch",
        ):
            X_group = self.X[np.array(row["X_index"])]

            if row[self.CONTROL_COL]:
                # collect controls by context and batch
                if X_group.shape[0] > 1:
                    self.baseline_ctrls[row[self.CONTEXT_COL]][row[self.BATCH_COL]] += (
                        X_group.mean(axis=0)
                    )
                else:
                    self.baseline_ctrls[row[self.CONTEXT_COL]][row[self.BATCH_COL]] += (
                        X_group.squeeze()
                    )
                self.baseline_ctrls_count[row[self.CONTEXT_COL]][
                    row[self.BATCH_COL]
                ] += 1
            else:
                # collect perturbations by context and batch
                pert_id = from_perturbations_to_compound_conc(row[self.PERT_COL])
                if X_group.shape[0] > 1:
                    self.baseline_perts[row[self.CONTEXT_COL]][pert_id][
                        row[self.BATCH_COL]
                    ] += X_group.mean(axis=0)
                else:
                    self.baseline_perts[row[self.CONTEXT_COL]][pert_id][
                        row[self.BATCH_COL]
                    ] += X_group.squeeze()
                self.baseline_perts_count[row[self.CONTEXT_COL]][pert_id][
                    row[self.BATCH_COL]
                ] += 1

        # get mean control and delta by context and perturbation
        baseline_ctrls_tmp = {}
        baseline_perts_tmp = defaultdict(lambda: defaultdict(dict))
        baseline_deltas_tmp = defaultdict(dict)
        for context, batch_pert_dict in self.baseline_perts.items():
            # 1. get mean control for current context
            control_mean = np.zeros(self.feature_shape)
            count = 0

            # collect control means for current context from all batches
            # aggregation is balanced (mean of mean per batch), so unbiased by batch size
            for batch, ctrls in self.baseline_ctrls[context].items():
                agg_count = self.baseline_ctrls_count[context][batch]
                control_mean += ctrls / agg_count
                count += 1

            if count > 0:
                control_mean /= count
                baseline_ctrls_tmp[context] = control_mean
            else:
                logger.warning(f"No controls found for {context}")
                baseline_ctrls_tmp[context] = np.zeros(self.feature_shape)

            # 2. get mean delta for current context and perturbation
            # iterate over perturbations
            for pert, pert_batches in batch_pert_dict.items():
                delta = np.zeros(self.feature_shape)

                # iterate over batches
                for batch, perts in pert_batches.items():
                    agg_count = self.baseline_perts_count[context][pert][batch]
                    pert_mean = perts / agg_count

                    # update batch-mean (correct for multiple observations)
                    baseline_perts_tmp[context][pert][batch] = pert_mean

                    # caclulate delta
                    batch_ctrl_mean = (
                        self.baseline_ctrls[context][batch]
                        / self.baseline_ctrls_count[context][batch]
                    )
                    delta += pert_mean - batch_ctrl_mean

                if len(pert_batches) > 0:
                    delta /= len(pert_batches)
                else:
                    delta = np.zeros(self.feature_shape)
                    logger.warning(f"No perturbations found for {context} {pert}")
                baseline_deltas_tmp[context][pert] = delta

        # dict of mean control for each context
        self.baseline_ctrls = baseline_ctrls_tmp
        self.baseline_perts = baseline_perts_tmp
        # dict of mean delta for each context and perturbation
        # to be aggregated by context OR perturbation depending on the baseline
        self.aggregated_deltas = baseline_deltas_tmp

    def fit_controls(self, test_dataset):
        """
        Get context controls means from held-out test dataset.

        This is only required for fully OOD splits.
        """
        self.test_obs = test_dataset.obs.with_row_index("X_index")
        self.test_obs = self.test_obs.filter(pl.col(self.CONTROL_COL))
        self.test_obs = self.test_obs.group_by(
            self.CONTEXT_COL, self.BATCH_COL, maintain_order=True
        ).agg(pl.col("X_index"))

        self.test_ctrls = defaultdict(
            lambda: defaultdict(partial(np.zeros, shape=(self.feature_shape,)))
        )
        for row in tqdm(
            self.test_obs.iter_rows(named=True),
            total=len(self.test_obs),
            desc="Fitting controls",
        ):
            X_group = self.X[np.array(row["X_index"])]
            self.test_ctrls[row[self.CONTEXT_COL]][row[self.BATCH_COL]] += X_group.mean(
                axis=0
            )


class ContextMeanBaseline(Baseline):
    """
    Mean across all training perturbations seen in target biological context.
    """

    def __init__(self, dataset, valid_indices):
        super().__init__(dataset, valid_indices)

        self._calculate_baseline()

    def _calculate_baseline(self):
        """
        Calculate baseline by averaging across seen perturbations for each context.
        """
        # aggregate deltas across perturbations for each context
        self.baseline_deltas = defaultdict(partial(np.zeros, shape=self.feature_shape))
        for context, delta_dict in self.aggregated_deltas.items():
            for pert, delta in delta_dict.items():
                self.baseline_deltas[context] += delta

            self.baseline_deltas[context] /= len(delta_dict)

    def forward(self, obs_row, is_drugscreen: bool = True):
        if self.baseline_deltas is None:
            raise ValueError("Baseline has to be calculated first.")

        if is_drugscreen:
            # context is the disease model (genetic perturbation)
            align_column = "plate_disease_model"
        else:
            align_column = self.CONTEXT_COL
        biological_context = obs_row[align_column]

        # get batch control mean
        if biological_context in self.baseline_ctrls.keys():
            baseline_pred = self.baseline_ctrls[biological_context].copy()
        else:
            raise ValueError(f"No controls found for {biological_context} context")

        # apply baseline delta to control
        if not obs_row[self.CONTROL_COL]:
            baseline_delta = self.baseline_deltas[biological_context].copy()
            baseline_pred += baseline_delta

        return baseline_pred


class PerturbationMeanBaseline(Baseline):
    """
    Mean across all occurences of target perturbation across training biological contexts.
    """

    def __init__(self, dataset, valid_indices):
        super().__init__(dataset, valid_indices)

        self._calculate_baseline()

    def _calculate_baseline(self):
        """
        Calculate baseline by averaging across occurences of target perturbation for each context.
        """
        # aggregate deltas across contexts for each perturbation
        self.baseline_deltas = defaultdict(partial(np.zeros, shape=self.feature_shape))
        for context, delta_dict in self.aggregated_deltas.items():
            for pert, delta in delta_dict.items():
                self.baseline_deltas[pert] += delta

            self.baseline_deltas[pert] /= len(delta_dict)

    def forward(self, obs_row, is_drugscreen: bool = True):
        if self.baseline_deltas is None:
            raise ValueError("Baseline has to be calculated first.")

        if is_drugscreen:
            # context is the disease model (genetic perturbation)
            context_column = "plate_disease_model"
        else:
            context_column = self.CONTEXT_COL
        biological_context = obs_row[context_column]

        # get batch control mean
        if biological_context in self.baseline_ctrls.keys():
            baseline_pred = self.baseline_ctrls[biological_context].copy()
        else:
            raise ValueError(f"No controls found for {biological_context} context")

        # apply baseline delta to control
        if not obs_row[self.CONTROL_COL]:
            if is_drugscreen:
                pert_id = from_perturbations_to_compound_conc(obs_row[self.PERT_COL])
            else:
                pert_id = obs_row[self.PERT_COL]
            baseline_delta = self.baseline_deltas[pert_id].copy()
            baseline_pred += baseline_delta

        return baseline_pred


class ContextSampleBaseline(Baseline):
    """
    Sample cells from target context from training dataset.

    NOTE: Implementation choice has yet to be finalized.
    Current implementation: sample cell from target context and random perturbation/batch.

    """

    def __init__(self, dataset, valid_indices):
        super().__init__(dataset, valid_indices)

    def forward(self, obs_row, is_drugscreen: bool = True):
        if is_drugscreen:
            # context is the disease model (genetic perturbation)
            align_column = "plate_disease_model"
        else:
            align_column = self.CONTEXT_COL
        biological_context = obs_row[align_column]

        # sample perturbation
        perts = self.baseline_perts[biological_context].keys()
        sampled_pert = list(perts)[np.random.choice(len(perts))]

        # sample batch
        batches = self.baseline_perts[biological_context][sampled_pert].keys()
        sampled_batch = list(batches)[np.random.choice(len(batches))]

        # sample cell from matched context
        sampled_cell = self.baseline_perts[biological_context][sampled_pert][
            sampled_batch
        ]

        return sampled_cell


class PerturbationSampleBaseline(Baseline):
    """
    Sample cells from target perturbation from training dataset.

    NOTE: Implementation choice has yet to be finalized.
    Current implementation: sample cell from target perturbation and random context/batch.

    """

    def __init__(self, dataset, valid_indices):
        super().__init__(dataset, valid_indices)

        raise NotImplementedError(
            "PerturbationSampleBaseline is not (fully) implemented yet."
        )

    def forward(self, obs_row):
        pert_id = from_perturbations_to_compound_conc(obs_row[self.PERT_COL])

        # sample perturbation
        contexts = self.baseline_perts.keys()
        sampled_context = np.random.choice(list(contexts))

        # sample batch
        batches = self.baseline_perts[sampled_context][pert_id].keys()
        sampled_batch = np.random.choice(list(batches))

        # sample cell from matched context
        sampled_cell = self.baseline_perts[sampled_context][pert_id][sampled_batch]

        return sampled_cell


class ExperimentalReproducibilityBaseline(Baseline):
    """ """

    def __init__(self, dataset, valid_indices):
        super().__init__(dataset, valid_indices)

        raise NotImplementedError(
            "ExperimentalReproducibilityBaseline is not implemented yet."
        )
