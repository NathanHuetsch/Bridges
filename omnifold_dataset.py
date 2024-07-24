import torch
import numpy as np


class Omnifold:

    def __init__(self, params):
        self.params = params

        self.path = params["path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()
        self.apply_preprocessing()
        self.init_observables()

    def init_dataset(self):
        try:
            dataset = np.load(self.path)
        except:
            dataset = np.load(self.params["path2"])
        n_data = self.params["n_data"]
        self.gen = torch.tensor(dataset["pythia_gen"][:n_data]).float().to(self.device)
        self.rec = torch.tensor(dataset["pythia_rec"][:n_data]).float().to(self.device)

    def apply_preprocessing(self, reverse=False):

        if not reverse:
            #self.gen = self.gen[:, :5]
            #self.rec = self.rec[:, :5]

            # add noise to the jet multiplicity to smear out the integer structure
            noise = torch.rand_like(self.rec[:, 1]) - 0.5
            self.rec[:, 1] = self.rec[:, 1] + noise
            noise = torch.rand_like(self.gen[:, 1]) - 0.5
            self.gen[:, 1] = self.gen[:, 1] + noise

            # standardize events
            self.rec_mean = self.rec.mean(0)
            self.rec_std = self.rec.std(0)
            self.gen_mean = self.gen.mean(0)
            self.gen_std = self.gen.std(0)

            self.gen = (self.gen - self.gen_mean)/self.gen_std
            self.rec = (self.rec - self.rec_mean)/self.rec_std

        else:
            if not hasattr(self, "rec_mean"):
                raise ValueError("Trying to run reverse preprocessing before forward preprocessing")

            # undo standardization
            self.gen = self.gen * self.gen_std + self.gen_mean
            self.rec = self.rec * self.rec_std + self.rec_mean

            # round jet multiplicity back to integers
            self.rec[:, 1] = torch.round(self.rec[:, 1])
            self.gen[:, 1] = torch.round(self.gen[:, 1])

            if hasattr(self, "unfolded"):
                self.unfolded = self.unfolded * self.gen_std + self.gen_mean
                self.unfolded[:, 1] = torch.round(self.unfolded[:, 1])

            if hasattr(self, "single_event_unfolded"):
                self.single_event_unfolded = self.single_event_unfolded * self.gen_std + self.gen_mean
                self.single_event_unfolded[..., 1] = torch.round(self.single_event_unfolded[..., 1])

    def init_observables(self):
        self.observables = []

        self.observables.append({
            "tex_label": r"\text{Jet mass } m",
            "bins": torch.linspace(1, 60, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Jet multiplicity } N",
            "bins": torch.arange(3.5, 60.5),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\text{N-subjettiness ratio } \tau_{21}$",
            "bins": torch.linspace(0.1, 1.1, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Jet width } w",
            "bins": torch.linspace(0, 0.6, 50 + 1),
            "yscale": "log"
        })
        self.observables.append({
            "tex_label": r"$\text{Groomed mass }\log \rho$",
            "bins": torch.linspace(-14, -2, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"$\text{Groomed momentum fraction }z_g$",
            "bins": torch.linspace(0.05, 0.55, 50 + 1),
            "yscale": "log"
        })








