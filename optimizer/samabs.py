import torch


class SAMABS(torch.optim.Optimizer):
    def __init__(self, params, rho=0.05, adaptive=False, group="B", condition=1, threshold=0.5, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAMABS, self).__init__(params, defaults)
        self.group = group
        self.condition = condition
        self.threshold = threshold

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.first_grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (self.first_grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]

                e_w = (torch.pow(p, 2)
                       if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

                param_state['first_grad'] = p.grad.clone()
                param_state['e_w'] = e_w.clone()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            step_size = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                p.sub_(param_state['e_w'])  # get back to "w" from "w + e(w)"

                ratio = p.grad.div(param_state['first_grad'].add(1e-8))
                if self.group == "A":
                    mask = ratio >= 1
                elif self.group == "B1":
                    mask = torch.logical_and(ratio > 0, ratio < self.threshold)
                elif self.group == "B2":
                    mask = torch.logical_and(ratio >= self.threshold, ratio < 1)
                elif self.group == "C":
                    mask = ratio <= 0

                d_p = p.grad.mul(mask).mul(self.condition) + p.grad.mul(torch.logical_not(mask))
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                if 'exp_avg' not in param_state:
                    param_state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                param_state['exp_avg'].mul_(momentum).add_(d_p)

                p.add_(param_state['exp_avg'], alpha=-step_size)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        # the closure should do a full forward-backward pass
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    @torch.no_grad()
    def _grad_norm(self, by=None):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        if by is None:
            norm = torch.norm(
                torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0)
                             * p.grad).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                            ]),
                p=2
            )
            return norm
        else:
            norm = torch.norm(
                torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0)
                             * self.state[p][by]).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                            ]),
                p=2
            )
            return norm

    @torch.no_grad()
    def _weight_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                        p.data.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                        ]),
            p=2
        )
        return norm

    def set_alpha(self, alpha):
        self.condition = alpha

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
