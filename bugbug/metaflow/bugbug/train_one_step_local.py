from metaflow import FlowSpec, step, conda_base, conda


from scripts.trainer import Trainer, parse_args


def get_python_version():
    """
    A convenience function to get the python version used to run this
    tutorial. This ensures that the conda environment is created with an
    available version of python.

    """
    import platform

    versions = {"3": "3.10.4"}
    return versions[platform.python_version_tuple()[0]]


class SpamBugFlow(FlowSpec):
    """
    A flow to train SpamBug.

    The flow performs the following steps:
    1) Uses the original CLI class to perform training.
    """

    @step
    def start(self):
        """
        Not doing anything
        """

        self.next(self.train)

    @step
    def train(self):
        """
        This step trains the BugBug spambug model

        """
        retriever = Trainer()
        args = parse_args(["spambug"])
        retriever.go(args)

        self.next(self.end)

    @step
    def end(self):
        """
        Finished training
        """
        print("Training completed for BugBug spambug")


if __name__ == "__main__":
    SpamBugFlow()
