"""Enable ``torchrun -m speculators.train config.yaml``."""

from speculators.train.cli import main
from speculators.train.config import TrainConfig

if __name__ == "__main__":
    main(TrainConfig.resolve())
