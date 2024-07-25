import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}'
FONTSIZE=16
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def marginal_plots(path, dataset):
    dims = dataset.gen.shape[-1]
    with PdfPages(path) as pp:
        for dim in range(dims):
            bins = dataset.observables[dim]["bins"]
            hist_rec, _ = np.histogram(dataset.rec[:, dim].cpu(), density=True, bins=bins)
            hist_gen, _ = np.histogram(dataset.gen[:, dim].cpu(), density=True, bins=bins)
            hist_unfolded, _ = np.histogram(dataset.unfolded[:, dim].cpu(), density=True, bins=bins)

            fig1, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.00})
            fig1.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.6, rect=(0.07, 0.06, 0.99, 0.95))

            axs[0].step(bins[1:], hist_rec, label="Rec", linewidth=1.0, where="post")
            axs[0].step(bins[1:], hist_gen, label="Gen", linewidth=1.0, where="post")
            axs[0].step(bins[1:], hist_unfolded, label="Unfolded", linewidth=1.0, where="post")
            axs[0].set_yscale(dataset.observables[dim]["yscale"])

            axs[1].step(bins[1:], hist_unfolded/hist_gen)

            axs[0].legend(frameon=False, fontsize=FONTSIZE)
            axs[0].set_ylabel("Normalized", fontsize=FONTSIZE)

            axs[1].set_ylabel(r"$\frac{\mathrm{CFM}}{\mathrm{True}}$", fontsize=FONTSIZE)
            axs[1].set_yticks([0.8,1,1.2])
            axs[1].set_ylim([0.71, 1.29])
            axs[1].axhline(y=1., c="black", ls="--", lw=0.7)
            axs[1].axhline(y=1.2, c="black", ls="dotted", lw=0.5)
            axs[1].axhline(y=0.8, c="black", ls="dotted", lw=0.5)

            axs[0].tick_params(axis="both", labelsize=FONTSIZE)
            axs[1].tick_params(axis="both", labelsize=FONTSIZE)

            plt.xlabel(dataset.observables[dim]["tex_label"], fontsize=FONTSIZE)
            plt.savefig(pp, format="pdf", bbox_inches="tight", pad_inches=0.05)
            plt.close()


def migration_plots(path, dataset):
    dims = dataset.gen.shape[-1]
    with PdfPages(path) as pp:
        for dim in range(dims):
            bins = dataset.observables[dim]["bins"]
            fig, axs = plt.subplots(1,2, figsize=(10,4))
            axs[0].hist2d(dataset.gen[:, dim].cpu().numpy(), dataset.rec[:, dim].cpu().numpy(), density=True, bins=bins, rasterized=True)
            axs[0].set_title(f"True " + dataset.observables[dim]["tex_label"], fontsize=FONTSIZE)
            axs[0].set_xlabel("Rec", fontsize=FONTSIZE)
            axs[0].set_ylabel("Gen", fontsize=FONTSIZE)

            axs[1].hist2d(dataset.unfolded[:, dim].cpu().numpy(), dataset.rec[:, dim].cpu().numpy(), density=True, bins=bins, rasterized=True)
            axs[1].set_title(f"Model " + dataset.observables[dim]["tex_label"], fontsize=FONTSIZE)
            axs[1].set_xlabel("Rec", fontsize=FONTSIZE)
            axs[1].set_ylabel("Unfold", fontsize=FONTSIZE)
            plt.savefig(pp, format="pdf", bbox_inches="tight")
            plt.close()


def single_event_plots(path, dataset):
    n_events = dataset.single_event_unfolded.shape[0]
    n_unfoldings = dataset.single_event_unfolded.shape[1]
    dims = dataset.single_event_unfolded.shape[2]

    with PdfPages(path) as pp:
        for dim in range(dims):
            for event in range(n_events):
                event_rec = dataset.rec[event, dim]
                event_gen = dataset.gen[event, dim]
                event_unfoldings = dataset.single_event_unfolded[event, :, dim]

                bins = dataset.observables[dim]["bins"]
                plt.axvline(event_rec.cpu(), label="Event Rec", color="blue")
                plt.axvline(event_gen.cpu(), label="Event Gen", color="orange")
                plt.hist(event_unfoldings.cpu(), density=True, histtype="step", bins=bins, label="Unfoldings", color="green")
                plt.hist(dataset.gen[:, dim].cpu(), density=True, bins=bins, label="Full Gen", alpha=0.2)
                plt.legend()
                plt.title(f"Event {event}", fontsize=FONTSIZE)
                plt.ylabel("Normalized", fontsize=FONTSIZE)
                plt.xlabel(dataset.observables[dim]["tex_label"], fontsize=FONTSIZE)
                plt.savefig(pp, format="pdf", bbox_inches="tight")
                plt.close()


def get_quantile_bins(data, n_bins=50, lower=0.001, upper=0.001):
    return torch.linspace(
        torch.nanquantile(data, lower), torch.nanquantile(data, 1 - upper), n_bins + 1
    )