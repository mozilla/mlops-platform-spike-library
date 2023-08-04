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


# The list of packages was pulled from requirements.txt.  Some of the packages are unavailable on
# conda-forge, or are under different names.  Some versions are unavailable.  Original values are
# included in comments.  For any packages removed, code using those packages has been commented out.
@conda_base(python=get_python_version(), libraries={
    "amqp": "5.1.1",
    "beautifulsoup4": "4.12.2",
    "boto3": "1.28.12",
    "google-auth": "2.22.0",
    "google-cloud-storage": "2.10.0",
    "imbalanced-learn": "0.11.0",
    "python-kubernetes": "27.2.0",
    # "libmozdata": "0.2.4",
    # replaced by python-lmdb "lmdb": "1.4.1",
    "python-lmdb": "1.4.1",
    "markdown2": "2.4.10",
    "matplotlib": "3.7.2",
    "mercurial": "6.4.5",  # was 6.5
    "metaflow": "2.9.11",
    # "microannotate": "0.0.24",
    # "mozci": "2.3.2",
    "notebook": "7.0.1",
    "numpy": "1.22.4",
    "orjson": "3.9.2",
    # "ortools": "9.6.2534",
    "pandas": "2.0.3",
    "plotly": "5.15.0",
    "psutil": "5.9.5",
    # "pydriller": "1.12",
    "pyOpenSSL": "23.2.0", # was pyOpenSSL>=0.14 
    # Could not find a version that satisfies the requirement pyOpenSSL>=0.14; extra ": " "security" (from requests[security]>=2.7.0->libmozdata": "0.1.43)
    "python-dateutil": "2.8.2",
    # "python-hglib": "2.6.2",
    "ratelimit": "2.2.1",
    "requests": "2.31.0",
    "scikit-learn": "1.1.3",
    "scipy": "1.11.1",
    "sendgrid": "6.0.5",  # was 6.10.0
    "shap[plots]": "0.42.1",
    "tabulate": "0.9.0",
    # "taskcluster": "54.4.1",
    "tenacity": "8.2.2",
    "tqdm": "4.65.0",
    "xgboost": "1.7.4",
    "zstandard": "0.19.0",  # was 0.21.0
})
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
