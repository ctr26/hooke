import typer

from vcb._cli.evaluate.px import evaluate_cli as evaluate_cli_px
from vcb._cli.evaluate.tx import evaluate_cli as evaluate_cli_tx
from vcb._cli.finetune import finetune

# CLI entrypoint
app = typer.Typer(pretty_exceptions_enable=False)
app.command("finetune")(finetune)

# Subapp: Evaluate
evaluate_app = typer.Typer(pretty_exceptions_enable=False)
evaluate_app.command("tx")(evaluate_cli_tx)
evaluate_app.command("px")(evaluate_cli_px)
app.add_typer(evaluate_app, name="evaluate")

if __name__ == "__main__":
    app()
