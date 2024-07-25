import argparse
import yaml
import os
import torch
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from omnifold_dataset import Omnifold
from models import CFM, Didi
from plots import marginal_plots, migration_plots, single_event_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    # read in the parameters
    with open(args.path, 'r') as f:
        params = yaml.safe_load(f)

    # create a results dir and save parameters to it
    dir_path = os.path.dirname(os.path.realpath(__file__))
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + params["run_name"]
    run_dir = os.path.join(dir_path, "results", run_name)
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "params.yaml"), 'w') as f:
        yaml.dump(params, f)

    # look for GPUs
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"Using device {device}")

    # get the dataset
    print("Building dataset", params["dataset_params"]["type"])
    dataset = eval(params["dataset_params"]["type"])(params["dataset_params"])
    # get gen and rec dimension from the dataset
    dims_gen = dataset.gen.shape[-1]
    dims_rec = dataset.rec.shape[-1]

    # build the model
    print("Building model", params["model_params"]["type"])
    model = eval(params["model_params"]["type"])(dims_gen, dims_rec, params["model_params"]).to(device)

    # train the model
    print("Training model")
    model.train(dataset.gen, dataset.rec, None)

    print("Saving model")
    os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)
    model_path = os.path.join(run_dir, "model", f"model.pth")
    torch.save(model.state_dict(), model_path)

    # evaluate the model
    print("Sampling model")
    dataset.unfolded = model.evaluate(dataset.rec)
    print("Sampling single event unfoldings")
    dataset.single_event_unfolded = model.single_event_unfolding(dataset.rec[:5])
    dataset.apply_preprocessing(reverse=True)

    # make plots
    print("Making plots")
    file_marginalplots = os.path.join(run_dir, f"plots_marginals.pdf")
    marginal_plots(file_marginalplots, dataset)
    file_migrationplots = os.path.join(run_dir, f"plots_migration.pdf")
    migration_plots(file_migrationplots, dataset)
    file_singleeventplots = os.path.join(run_dir, f"plots_singleevent.pdf")
    single_event_plots(file_singleeventplots, dataset)

if __name__ == '__main__':
    main()