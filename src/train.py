from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.plugins.environments import SLURMEnvironment

# fmt: off
# isort: off
from model import LightningClassifierModel
from data import LightningTraumaData
from util import StrictWandbLogger
# isort: on
# fmt: on


def run():
    cli = LightningCLI()


if __name__ == "__main__":
    run()
