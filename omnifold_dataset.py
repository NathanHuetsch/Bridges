import torch
import numpy as np


class Omnifold:

    def __init__(self, params):
        self.params = params

        self.path = params["path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()
        self.apply_preprocessing()

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

            self.gen = self.gen[:, :5]
            self.rec = self.rec[:, :5]
            
        else:
            if not hasattr(self, "rec_mean"):
                raise ValueError("Trying to run reverse preprocessing before forward preprocessing")

            # undo standardization
            self.gen = self.gen * self.gen_std + self.gen_mean
            self.rec = self.rec * self.rec_std + self.rec_mean

            # round jet multiplicity back to integers
            self.rec[:, 1] = torch.round(self.rec[:, 1])
            self.gen[:, 1] = torch.round(self.gen[:, 1])
