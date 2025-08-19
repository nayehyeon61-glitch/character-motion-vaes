import os
import math
import torch
import torch.nn as nn
import torch.optim as optim

class PPO(object):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        mirror_function=None,
        # ---- New: MaxEnt (Entropy-PPO) options ----
        adaptive_entropy: bool = False,
        target_entropy: float = None,
        entropy_lr: float = None,           # e.g., 3e-4
        init_entropy_coef: float = None,    # if None -> use entropy_coef
    ):
        self.mirror_function = mirror_function
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef     # used when adaptive_entropy=False
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        # -------- Entropy auto-tuning (SAC-style) ----------
        self.adaptive_entropy = adaptive_entropy

        if self.adaptive_entropy:
            # try to infer action_dim for a decent default target entropy
            if target_entropy is None:
                # Best-effort: pull from policy distribution if available
                try:
                    # a quick dummy forward to get action shape may not be feasible here;
                    # fall back to actor_critic attribute if it exists
                    action_dim = getattr(actor_critic, "action_dim", None)
                    if action_dim is None and hasattr(actor_critic, "dist_action_dim"):
                        action_dim = actor_critic.dist_action_dim
                    if action_dim is None:
                        # final fallback
                        action_dim = 1
                except Exception:
                    action_dim = 1
                target_entropy = -float(action_dim)

            self.target_entropy = float(target_entropy)

            # alpha (temperature) parameterized via log_alpha
            init_alpha = init_entropy_coef if init_entropy_coef is not None else entropy_coef
            init_alpha = max(1e-8, float(init_alpha))
            device = next(actor_critic.parameters()).device
            self.log_alpha = torch.tensor(math.log(init_alpha), requires_grad=True, device=device)

            if entropy_lr is None:
                entropy_lr = lr if lr is not None else 3e-4
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=entropy_lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None

    @property
    def alpha(self):
        if self.adaptive_entropy:
            return self.log_alpha.exp()
        else:
            # fixed coefficient mode
            return torch.tensor(self.entropy_coef, device=next(self.actor_critic.parameters()).device)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        alpha_value_epoch = 0.0  # for logging

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                if self.mirror_function is not None:
                    (
                        observations_batch,
                        actions_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ,
                    ) = self.mirror_function(sample)
                else:
                    (
                        observations_batch,
                        actions_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ,
                    ) = sample

                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
                    observations_batch, actions_batch
                )

                # --- PPO clipped objective ---
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # --- Critic loss (standard) ---
                value_loss = (return_batch - values).pow(2).mean()

                # --- Build total loss ---
                # If adaptive_entropy: use alpha * E[log pi(a|s)] term (== -alpha * H)
                # Else: keep original entropy bonus path for full backward compatibility
                if self.adaptive_entropy:
                    alpha = self.alpha.detach()  # treat alpha constant in actor update
                    entropy_term = (alpha * action_log_probs).mean()  # (+) because log_probs <= 0
                    total_loss = value_loss * self.value_loss_coef + action_loss + entropy_term
                else:
                    total_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                # --- Backprop (policy+value) ---
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # --- If adaptive, update alpha (temperature) to match target entropy ---
                if self.adaptive_entropy:
                    # Use sample-wise -log pi as an unbiased estimator of entropy
                    # Note: detach to avoid affecting the policy during alpha update
                    with torch.no_grad():
                        # negative log prob ~ per-sample entropy contribution
                        neg_logp = (-action_log_probs).detach()

                    alpha_loss = -(self.log_alpha * (neg_logp - self.target_entropy)).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    alpha_value_epoch += float(self.alpha.item())

                value_loss_epoch += float(value_loss.item())
                action_loss_epoch += float(action_loss.item())
                dist_entropy_epoch += float(dist_entropy.item())

        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        if self.adaptive_entropy:
            alpha_value_epoch /= num_updates
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, alpha_value_epoch
        else:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
