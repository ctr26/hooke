import typer

from vcb.evaluate.dataloader import DrugscreenDataloader
from vcb.models.dataset import Dataset, DatasetPaths
from vcb.models.split import Split

app = typer.Typer()


@app.command()
def hello(root: str, split_path: str):
    # NOTE (cwognum): I'm using this CLI as a scrappy test bed. Don't judge me.

    dataset = Dataset(paths=DatasetPaths(root=root))
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


if __name__ == "__main__":
    app()
