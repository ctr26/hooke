from vcb.data.dataloader import DrugscreenDataloader
from vcb.models.dataset import Dataset, DatasetDirectory
from vcb.models.split import Split


def finetune(root: str, split_path: str):
    """Finetune a baseline model."""

    # NOTE (cwognum): I'm using this CLI as a scrappy test bed. Don't judge me.

    dataset = Dataset(**DatasetDirectory(root=root).model_dump())
    split = Split.from_json(split_path)

    for fold in split.folds:
        # Finetune your model here...
        dataloader = DrugscreenDataloader(dataset=dataset, indices=fold.finetune)
        (control, base, perturbed, perturbations, biological_context) = dataloader[0]
        print(
            control.shape,
            base.shape,
            perturbed.shape,
            perturbations,
            biological_context,
        )

        # Evaluate your model here...
        dataloader = DrugscreenDataloader(dataset=dataset, indices=fold.test)
        (control, base, perturbed, perturbations, biological_context) = dataloader[0]
        print(
            control.shape,
            base.shape,
            perturbed.shape,
            perturbations,
            biological_context,
        )

        break
