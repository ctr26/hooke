import typer

from vcb._cli.evaluate.baseline_px_cli import run_baseline_px_cli
from vcb._cli.evaluate.baseline_tx_cli import run_baseline_tx_cli
from vcb._cli.evaluate.px_cli import px_evaluate_cli
from vcb._cli.evaluate.tx_cli import tx_evaluate_cli
from vcb._cli.split.drugscreen import drugscreen_split_cli

# CLI entrypoint
app = typer.Typer(pretty_exceptions_enable=False)

# Subapp: predictions
predictions_app = typer.Typer(pretty_exceptions_enable=False)
predictions_app.command("tx")(tx_evaluate_cli)
predictions_app.command("px")(px_evaluate_cli)
app.add_typer(predictions_app, name="predictions")

# Subapp: baselines
baseline_app = typer.Typer(pretty_exceptions_enable=False)
baseline_app.command("tx")(run_baseline_tx_cli)
baseline_app.command("px")(run_baseline_px_cli)
app.add_typer(baseline_app, name="baseline")

# Subapp: Evaluate
evaluate_app = typer.Typer(pretty_exceptions_enable=False)
evaluate_app.add_typer(predictions_app, name="predictions")
evaluate_app.add_typer(baseline_app, name="baseline")

# Subapp: Split
split_app = typer.Typer(pretty_exceptions_enable=False)
split_app.command("drugscreen")(drugscreen_split_cli)
app.add_typer(split_app, name="split")

if __name__ == "__main__":
    app()
