"""
Copyright (c) 2023 Saeejith Nair, University of Waterloo.
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import tyro
from nerfstudio.configs.config_utils import convert_markup_to_ansi
import wandb
from elastic_nerf.nerfacc.utils import (
    set_random_seed,
)
from elastic_nerf.nerfacc.trainers.ngp_occ import NGPOccTrainer, NGPOccTrainerConfig

set_random_seed(42)


def main(config: NGPOccTrainerConfig):
    trainer: NGPOccTrainer = config.setup()
    error_occurred = False  # Track if an error occurred

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Interrupted by user, finishing up...")
        error_occurred = True  # Update the flag
        raise  # Re-raise the KeyboardInterrupt exception
    except Exception as e:  # Catch other exceptions
        print(f"An error occurred: {e}")
        error_occurred = True  # Update the flag
        raise  # Re-raise the exception
    finally:
        exit_code = 1 if error_occurred else 0  # Determine the exit code
        wandb.finish(exit_code=exit_code)  # Pass the exit code


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
                tyro.conf.FlagConversionOff[NGPOccTrainerConfig]
            ],
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
