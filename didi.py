import torch
import torch.nn as nn
import time


class Didi(nn.Module):
    def __init__(self, dims_x, dims_c, params):
        super().__init__()
        self.dims_x = dims_x
        self.params = params
        self.cond_x1 = self.params.get("cond_x1", False)
        self.noise_scale = self.params.get("noise_scale", 1.e-2)

        self.init_network()

    def init_network(self):
        layers = []
        dims_in = self.dims_x + 1 if not self.cond_x1 else 2*self.dims_x + 1
        layers.append(nn.Linear(dims_in, self.params["internal_size"]))
        layers.append(nn.ReLU())
        for _ in range(self.params["hidden_layers"]):
            layers.append(nn.Linear(self.params["internal_size"], self.params["internal_size"]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.params["internal_size"], self.dims_x))
        self.network = nn.Sequential(*layers)

    def sample(self, x_1):
        dtype = x_1.dtype
        device = x_1.device

        def net_wrapper(t, x_t):
            t = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            if self.cond_x1:
                f = self.network(torch.cat([t, x_t, x_1], dim=-1))
            else:
                f = self.network(torch.cat([t, x_t], dim=-1))
            return f

        steps = torch.linspace(1, 0, self.params.get("n_steps", 1000))
        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = pair_steps
        x_t = x_1.detach()
        for tprev, t in pair_steps:
            drift = net_wrapper(t, x_t)
            pred_x0 = x_t - t * drift
            x_t = (t - tprev) / t * pred_x0 + tprev / t * x_t
            x_t += (self.noise_scale * tprev * (t - tprev) / t).sqrt() * torch.randn_like(x_t)
        return x_t

    def batch_loss(self, x_0, x_1, weight):
        noise = torch.randn_like(x_0)
        t = torch.rand((x_0.size(0), 1)).to(x_0.device)
        x_t = (1 - t) * x_0 + t * x_1 + (self.noise_scale*t*(1.-t)).sqrt() * noise
        f = (x_t-x_0)/t
        if self.cond_x1:
            f_pred = self.network(torch.cat([t, x_t, x_1], dim=-1))
        else:
            f_pred = self.network(torch.cat([t, x_t], dim=-1))
        cfm_loss = ((f_pred - f) ** 2 * weight.unsqueeze(-1)).mean()
        return cfm_loss

    def train(self, data_gen, data_rec, weights=None):
        if weights is None:
            weights = torch.ones((data_gen.shape[0]))
        dataset = torch.utils.data.TensorDataset(data_gen, data_rec, weights)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params["batch_size"],
                                             shuffle=True)
        n_epochs = self.params["n_epochs"]
        lr = self.params["lr"]
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader) * n_epochs)
        print(f"Training for {n_epochs} epochs with lr {lr}")
        t0 = time.time()
        for epoch in range(n_epochs):
            losses = []
            for i, batch in enumerate(loader):
                x_hard, x_reco, weight = batch
                optimizer.zero_grad()
                loss = self.batch_loss(x_hard, x_reco, weight)
                if loss < 100:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    losses.append(loss.item())
                else:
                    print(f"    Skipped update in epoch {epoch}, batch {i}, loss is", loss.item())
            if epoch % int(n_epochs / 5) == 0:
                print(f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")
        print(f"    Finished epoch {epoch} with average loss {torch.tensor(losses).mean()} after time {round(time.time() - t0, 1)}")

    def evaluate(self, data_c):
        predictions = []
        with torch.no_grad():
            batches = torch.split(data_c, self.params["batch_size_sample"])
            t0 = time.time()
            for i, batch in enumerate(batches):
                unfold = self.sample(batch).detach()
                predictions.append(unfold)
                t1 = time.time()
                if i == 0:
                    print(f"    Total batches: {len(batches)}. First batch took {round(t1 - t0, 1)} seconds")
            t2 = time.time()
            print(f"    Finished sampling after {round(t2 - t0, 1)} seconds")
        predictions = torch.cat(predictions)
        return predictions
