import torch
from torch.utils.data import DataLoader

class DLoader(DataLoader):
    def __init__ (self, dataset, batch_size=64, max_batch_size=1024, shuffle=False, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.start_iter = iter(self)
        self.current = 0
        self.batch_hist = []

        self.defaults = {
            "A": 0.0,
            "B": 0.0,
            "MaxB": max_batch_size,
            "k": 0,
            "b0": batch_size,
            "prev_len": 0,
            "next_len": batch_size,
            "len": batch_size,
            "beta": 0,
            "x": torch.Tensor([]),
            "grad": torch.Tensor([]),
            "prev_x": torch.Tensor([]),
            "prev_grad": torch.Tensor([]),
            "prev_beta": 0,
        }

    def step(self, params):
        x = torch.Tensor([])
        grad = torch.Tensor([])
        flag = True
        for p in params:
            if flag:
                flag = False
                x = torch.flatten(p.data)
                grad = torch.flatten(p.grad.data)
            else:
                x = torch.cat([x, torch.flatten(p.data)])

                grad = torch.cat([grad, torch.flatten(p.grad.data)])

        k = self.defaults["k"]
        if k == 0:
            self.defaults["grad"] = grad
            self.defaults["x"] = x
            self.defaults['k'] += 1
            self.defaults["next_len"] = min(self.defaults["len"] * 2, self.defaults["MaxB"])
            return

        self.defaults["prev_grad"] = self.defaults["grad"]
        self.defaults["prev_x"] = self.defaults["x"]
        self.defaults["grad"] = grad
        self.defaults["x"] = x
        beta = torch.dot(
            self.defaults["grad"] - self.defaults["prev_grad"],
            self.defaults["x"] - self.defaults["prev_x"]
        )
        self.defaults["len_prev"] = self.defaults["len"]
        self.defaults["len"] = self.defaults["next_len"]
        self.defaults["next_len"] = min(self.defaults["len"] * 2, self.defaults["MaxB"])
        self.defaults['beta'] = beta

        if k > 1:
            beta_prev = self.defaults["beta"]
            len_prev = self.defaults["len_prev"]
            len_curr = self.defaults["len"]

            if len_prev != len_curr:
                B = (beta_prev - beta) / (len_prev - len_curr)
                A = len_curr - beta * B
                self.defaults["A"] = A
                self.defaults["B"] = B

                next_len = max(1, min(self.defaults["MaxB"], self.defaults["B"] / (beta - A)))
                self.defaults["next_len"] = next_len

        self.defaults['k'] += 1

    def __iter__(self):
        self.start_iter = iter(super().iter())
        self.current = 0
        return self

    def __next__(self):
        if self.current < len(self):
            self.current += 1
            

            n = max(1, self.defaults["next_len"] // self.defaults["b0"])
            n = min(n, self.defaults["MaxB"] // self.defaults["b0"])
            self.batch_hist.append(self.defaults["b0"] * n)
            batches = []
            try:
                for _ in range(n):
                    batches.append(next(self.start_iter))
            except StopIteration:
                pass

            if len(batches) == 0:
                raise StopIteration

            first_batch = batches[0]
            for additional_batch in batches[1:]:
                for i, data in enumerate(additional_batch):
                    first_batch[i] = torch.cat([first_batch[i], data])

            return first_batch
        else:
            self.current = 0
            raise StopIteration
