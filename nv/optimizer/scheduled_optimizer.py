class ScheduledOptimizer():
    """ From https://github.com/jadore801120/attention-is-all-you-need-pytorch """

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        lr = (self.d_model ** -0.5) \
              * min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))
        return lr

    def update_learning_rate(self):
        self.n_steps += 1
        lr = self.lr_mul * self.get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
