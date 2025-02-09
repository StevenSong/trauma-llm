from lightning.pytorch.cli import LightningCLI

# fmt: off
# isort: off
from model import LightningClassifierModel
from data import LightningTraumaData
from util import StrictWandbLogger, PredictionWriter
# isort: on
# fmt: on


def run():
    cli = LightningCLI()


if __name__ == "__main__":
    run()
