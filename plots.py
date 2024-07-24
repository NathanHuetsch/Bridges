import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def marginal_plots(path, gen, rec, unfolded):
    dims = gen.shape[-1]
    with PdfPages(path) as pp:
        for dim in range(dims):
            bins = get_quantile_bins(rec[:, dim], n_bins=100)
            plt.hist(rec[:, dim].cpu(), density=True, histtype="step", bins=bins, label="Rec")
            plt.hist(gen[:, dim].cpu(), density=True, histtype="step", bins=bins, label="Gen")
            plt.hist(unfolded[:, dim].cpu(), density=True, histtype="step", bins=bins, label="Unfolded")
            plt.legend()
            plt.title(f"Unfolded Dim {dim}")
            plt.savefig(pp, format="pdf", bbox_inches="tight")
            plt.close()


def migration_plots(path, gen, rec, unfolded):
    dims = gen.shape[-1]
    with PdfPages(path) as pp:
        for dim in range(dims):
            bins = get_quantile_bins(rec[:, dim], n_bins=100)
            fig, axs = plt.subplots(1,2, figsize=(10,4))
            axs[0].hist2d(gen[:, dim].numpy(), rec[:, dim].numpy(), density=True, bins=bins)
            axs[0].set_title(f"Data migration Dim {dim}")

            axs[1].hist2d(unfolded[:, dim].numpy(), rec[:, dim].numpy(), density=True, bins=bins)
            axs[1].set_title(f"Model migration Dim {dim}")
            plt.savefig(pp, format="pdf", bbox_inches="tight")
            plt.close()


def get_quantile_bins(data, n_bins=50, lower=0.001, upper=0.001):
    return torch.linspace(
        torch.nanquantile(data, lower), torch.nanquantile(data, 1 - upper), n_bins + 1
    )