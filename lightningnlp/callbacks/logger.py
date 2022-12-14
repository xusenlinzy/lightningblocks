import os

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info


class LoggingCallback(pl.Callback):

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info(f"{key} = {str(metrics[key])}\n")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")

        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info(f"{key} = {str(metrics[key])}\n")
                    writer.write(f"{key} = {str(metrics[key])}\n")
