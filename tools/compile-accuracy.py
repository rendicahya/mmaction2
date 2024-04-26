import sys

sys.path.append(".")

import json
import pathlib

import click


@click.command()
@click.argument(
    "directory",
    nargs=1,
    required=True,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        path_type=pathlib.Path,
    ),
)
def main(directory):
    val = []
    test = []
    root = pathlib.Path.cwd()

    click.echo("\nReading train directory:")
    for var in (directory / "train").iterdir():
        click.echo(var.relative_to(root))

        pth_file = list(var.glob("best_acc_top1_epoch_*.pth"))[0]
        best_epoch = pth_file.stem.split("_")[-1]
        scalars_file = list(var.glob("**/scalars.json"))[0]
        pattern = f'"step": {best_epoch}}}'

        with open(scalars_file) as f:
            for line in f:
                if line.strip().startswith('{"acc/top1":') and line.strip().endswith(
                    f'"step": {best_epoch}}}'
                ):
                    json_data = json.loads(line)

                    val.append(f"{json_data['acc/top1']},{json_data['acc/top5']}")

                    break

    click.echo("\nValidation accuracies:")
    click.echo("\n".join(val))

    click.echo("\nReading test directory:")
    for var in (directory / "test").iterdir():
        click.echo(var.relative_to(root))

        json_file = list(var.glob("**/*.json"))[0]

        with open(json_file) as f:
            json_data = json.loads(f.read())
            test.append(f"{json_data['acc/top1']},{json_data['acc/top5']}")

    click.echo("\nTest accuracies:")
    click.echo("\n".join(test))


if __name__ == "__main__":
    main()
