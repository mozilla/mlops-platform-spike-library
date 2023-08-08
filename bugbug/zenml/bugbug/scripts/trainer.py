# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import os
import sys
from logging import INFO, basicConfig, getLogger
from zenml import pipeline, step

from bugbug import db
from bugbug.models import MODELS, get_model_class
from bugbug.utils import CustomJsonEncoder, zstd_compress

MODELS_WITH_TYPE = ("component",)

basicConfig(level=INFO)
logger = getLogger(__name__)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model_name = None
        self.model_obj = None
        self._get_model_object()

    def download_datasets(self):
        # Download datasets that were built by bugbug_data.
        os.makedirs("data", exist_ok=True)

        if self.args.download_db:
            for required_db in self.model_obj.training_dbs:
                assert db.download(required_db)

            if self.args.download_eval:
                self.model_obj.download_eval_dbs()
        else:
            logger.info("Skipping download of the databases")

    def _get_model_object(self):
        if self.args.classifier != "default":
            assert (
                self.args.model in MODELS_WITH_TYPE
            ), f"{self.args.classifier} is not a valid classifier type for {self.args.model}"

            self.model_name = f"{self.args.model}_{self.args.classifier}"
        else:
            self.model_name = self.args.model

        model_class = get_model_class(self.model_name)
        parameter_names = set(inspect.signature(model_class.__init__).parameters)
        parameters = {
            key: value for key, value in vars(self.args).items() if key in parameter_names
        }
        self.model_obj = model_class(**parameters)

    def go(self):
        logger.info("Training *%s* model", self.model_name)
        metrics = self.model_obj.train(limit=self.args.limit)

        # Save the metrics as a file that can be uploaded as an artifact.
        metric_file_path = "metrics.json"
        with open(metric_file_path, "w") as metric_file:
            json.dump(metrics, metric_file, cls=CustomJsonEncoder)

        logger.info("Training done")

        model_file_name = f"{self.model_name}model"
        assert os.path.exists(model_file_name)
        zstd_compress(model_file_name)

        logger.info("Model compressed")

        if self.model_obj.store_dataset:
            assert os.path.exists(f"{model_file_name}_data_X")
            zstd_compress(f"{model_file_name}_data_X")
            assert os.path.exists(f"{model_file_name}_data_y")
            zstd_compress(f"{model_file_name}_data_y")


def parse_args(args):
    description = "Train the models"
    main_parser = argparse.ArgumentParser(description=description)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--limit",
        type=int,
        help="Only train on a subset of the data, used mainly for integrations tests",
    )
    parser.add_argument(
        "--no-download",
        action="store_false",
        dest="download_db",
        help="Do not download databases, uses whatever is on disk",
    )
    parser.add_argument(
        "--download-eval",
        action="store_true",
        dest="download_eval",
        help="Download databases and database support files required at runtime (e.g. if the model performs custom evaluations)",
    )
    parser.add_argument(
        "--lemmatization",
        help="Perform lemmatization (using spaCy)",
        action="store_true",
    )
    parser.add_argument(
        "--classifier",
        help="Type of the classifier. Only used for component classification.",
        choices=["default", "nn"],
        default="default",
    )

    subparsers = main_parser.add_subparsers(title="model", dest="model", required=True)

    for model_name in MODELS:
        subparser = subparsers.add_parser(
            model_name, parents=[parser], help=f"Train {model_name} model"
        )

        try:
            model_class_init = get_model_class(model_name).__init__
        except ImportError:
            continue

        for parameter in inspect.signature(model_class_init).parameters.values():
            if parameter.name == "self":
                continue

            # Skip parameters handled by the base class (TODO: add them to the common argparser and skip them automatically without hardcoding by inspecting the base class)
            if parameter.name == "lemmatization":
                continue

            parameter_type = parameter.annotation
            if parameter_type == inspect._empty:
                parameter_type = type(parameter.default)
            assert parameter_type is not None

            if parameter_type == bool:
                subparser.add_argument(
                    f"--{parameter.name}"
                    if parameter.default is False
                    else f"--no-{parameter.name}",
                    action="store_true"
                    if parameter.default is False
                    else "store_false",
                    dest=parameter.name,
                )
            else:
                subparser.add_argument(
                    f"--{parameter.name}",
                    default=parameter.default,
                    dest=parameter.name,
                    type=int,
                )

    return main_parser.parse_args(args)

@step(enable_cache=False)
def get_trainer() -> Trainer:
    logger.info("Step: get_trainer")
    args = parse_args(sys.argv[1:])
    retriever = Trainer(args)
    return retriever

@step(enable_cache=False)
def download_datasets(retriever: Trainer) -> Trainer:
    logger.info("Step: download_datasets")
    retriever.download_datasets()
    return retriever

@step(enable_cache=False)
def train(retriever: Trainer):
    logger.info("Step: train")
    retriever.go()

@pipeline
def main():
    retriever = get_trainer()
    retriever = download_datasets(retriever)
    train(retriever)

if __name__ == "__main__":
    main()
