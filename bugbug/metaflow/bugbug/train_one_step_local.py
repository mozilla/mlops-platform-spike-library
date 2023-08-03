from metaflow import FlowSpec, step

from scripts.trainer import Trainer, parse_args


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
