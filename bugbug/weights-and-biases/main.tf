terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.51.0"
    }
  }
}

provider "google" {
  credentials = file("moz-fx-dev-ctroy-ml-ops-spikes-svc-acct-key.json")

  project = "moz-fx-dev-ctroy-ml-ops-spikes"
  region  = "us-central1"
  zone    = "us-central1-c"
}

resource "google_compute_network" "vpc_network" {
  name = "terraform-network"
}

module "wandb" {
  source    = "terraform-google-wandb"
  namespace = "mlops-wandb"
  license = "TEST.md"
}
