import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.controller import init, DiagGaussian


class AutoEncoder(nn.Module):
    def __init__(self, frame_size, latent_size, normalization):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        h1 = 256
        h2 = 128
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(frame_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(latent_size, h2)
        self.fc5 = nn.Linear(h2, h1)
        self.fc6 = nn.Linear(h1, frame_size)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

    def decode(self, x):
        h4 = F.relu(self.fc4(x))
        h5 = F.relu(self.fc5(h4))
        return self.fc6(h5)


class Encoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = frame_size * (num_future_predictions + num_condition_frames)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(frame_size + hidden_size, hidden_size)
        self.mu = nn.Linear(frame_size + hidden_size, latent_size)
        self.logvar = nn.Linear(frame_size + hidden_size, latent_size)

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Decoder
        # Takes latent | condition as input
        input_size = latent_size + frame_size * num_condition_frames
        output_size = num_future_predictions * frame_size
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.fc5 = nn.Linear(latent_size + hidden_size, hidden_size)
        self.out = nn.Linear(latent_size + hidden_size, output_size)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def forward(self, z, c):
        return self.decode(z, c)


class MixedDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c

        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out


class PoseMixtureVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 256
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder(*args)
        self.decoder = MixedDecoder(*args, num_experts)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def encode(self, x, c):
        _, mu, logvar = self.encoder(x, c)
        return mu, logvar

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return self.decoder(z, c), mu, logvar

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)


class PoseMixtureSpecialistVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 128
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder(*args)

        self.decoders = []
        for i in range(num_experts):
            decoder = Decoder(*args)
            self.decoders.append(decoder)
            self.add_module("d" + str(i), decoder)

        # Gating network
        gate_hsize = 128
        input_size = latent_size + frame_size * num_condition_frames
        self.g_fc1 = nn.Linear(input_size, gate_hsize)
        self.g_fc2 = nn.Linear(latent_size + gate_hsize, gate_hsize)
        self.g_fc3 = nn.Linear(latent_size + gate_hsize, num_experts)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def gate(self, z, c):
        h1 = F.elu(self.g_fc1(torch.cat((z, c), dim=1)))
        h2 = F.elu(self.g_fc2(torch.cat((z, h1), dim=1)))
        return self.g_fc3(torch.cat((z, h2), dim=1))

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        coefficients = F.softmax(self.gate(z, c), dim=1)
        predictions = torch.stack([decoder(z, c) for decoder in self.decoders], dim=1)
        return predictions, mu, logvar, coefficients

    def sample(self, z, c, deterministic=False):
        coefficients = F.softmax(self.gate(z, c), dim=1)
        predictions = torch.stack([decoder(z, c) for decoder in self.decoders], dim=1)

        if not deterministic:
            dist = torch.distributions.Categorical(coefficients)
            indices = dist.sample()
        else:
            indices = coefficients.argmax(dim=1)

        return predictions[torch.arange(predictions.size(0)), indices]


class PoseVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        h1 = 256
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(
            frame_size * (num_future_predictions + num_condition_frames), h1
        )
        self.fc2 = nn.Linear(frame_size + h1, h1)
        # self.fc3 = nn.Linear(h1, h1)
        self.mu = nn.Linear(frame_size + h1, latent_size)
        self.logvar = nn.Linear(frame_size + h1, latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(latent_size + frame_size * num_condition_frames, h1)
        self.fc5 = nn.Linear(latent_size + h1, h1)
        # self.fc6 = nn.Linear(latent_size + h1, h1)
        self.out = nn.Linear(latent_size + h1, num_future_predictions * frame_size)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        # h3 = F.elu(self.fc3(h2))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        # h6 = F.elu(self.fc6(torch.cat((z, h5), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, z, c, deterministic=False):
        return self.decode(z, c)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, latent_size):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.latent_size = latent_size

        # self.embedding = nn.Embedding(self.num_embeddings, self.latent_size)
        # self.embedding.weight.data.normal_()

        embed = torch.randn(latent_size, num_embeddings)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

        self.commitment_cost = 0.25
        self.decay = 0.99
        self.epsilon = 1e-5

    def forward(self, inputs):
        # Calculate distances
        dist = (
            inputs.pow(2).sum(1, keepdim=True)
            - 2 * inputs @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_embeddings).type(inputs.dtype)
        embed_ind = embed_ind.view(*inputs.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        # Use EMA to update the embedding vectors
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )

            embed_sum = inputs.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        loss = (quantize.detach() - inputs).pow(2).mean()
        quantize = inputs + (quantize - inputs).detach()

        avg_probs = embed_onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * (avg_probs + 1e-10).log()))

        return quantize, loss, perplexity, embed_ind


class PoseVQVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_embeddings,
        num_condition_frames,
        num_future_predictions,
        normalization,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        h1 = 512
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(
            frame_size * (num_future_predictions + num_condition_frames), h1
        )
        self.fc2 = nn.Linear(h1, h1)
        self.fc3 = nn.Linear(h1, h1)
        self.mu = nn.Linear(h1, latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(latent_size + frame_size * num_condition_frames, h1)
        self.fc5 = nn.Linear(h1, h1)
        self.fc6 = nn.Linear(h1, h1)
        self.out = nn.Linear(h1, num_future_predictions * frame_size)

        self.quantizer = VectorQuantizer(num_embeddings, latent_size)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def forward(self, x, c):
        mu = self.encode(x, c)
        quantized, loss, perplexity, _ = self.quantizer(mu)
        recon = self.decode(quantized, c)
        return recon, loss, perplexity

    def encode(self, x, c):
        s = torch.cat((x, c), dim=1)
        h1 = F.relu(self.fc1(s))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.mu(h3)

    def decode(self, z, c):
        s = torch.cat((z, c), dim=1)
        h4 = F.relu(self.fc4(s))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return self.out(h6)

    def sample(self, z, c, deterministic=False):
        if not deterministic:
            dist = torch.distributions.Categorical(z.softmax(dim=1))
            indices = dist.sample()
        else:
            indices = z.argmax(dim=1)
        z = F.embedding(indices, self.quantizer.embed.transpose(0, 1))
        s = torch.cat((z, c), dim=1)
        h4 = F.relu(self.fc4(s))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return self.out(h6)


class PoseVAEController(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        init_r_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        init_s_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("sigmoid"),
        )
        init_t_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("tanh"),
        )

        h_size = 256
        self.actor = nn.Sequential(
            init_r_(nn.Linear(self.observation_dim, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_t_(nn.Linear(h_size, self.action_dim)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.actor(x)


class PoseVAEPolicy(nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.actor = controller
        self.dist = DiagGaussian(controller.action_dim, controller.action_dim)

        init_s_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("sigmoid"),
        )
        init_r_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        h_size = 256
        self.critic = nn.Sequential(
            init_r_(nn.Linear(controller.observation_dim, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_r_(nn.Linear(h_size, h_size)),
            nn.ReLU(),
            init_s_(nn.Linear(h_size, 1)),
        )
        self.state_size = 1

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        action = self.actor(inputs)
        dist = self.dist(action)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
            action.clamp_(-1.0, 1.0)

        action_log_probs = dist.log_probs(action)
        value = self.critic(inputs)

        return value, action, action_log_probs

    def get_value(self, inputs):
        value = self.critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value = self.critic(inputs)
        mode = self.actor(inputs)
        dist = self.dist(mode)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
# ===== Diffusion for Pose Latent Space =====
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Sinusoidal time embedding (Transformer-style) ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t: (B,) with integer timesteps
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device).float() * (-math.log(10000.0) / (half - 1 + 1e-8))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb  # (B, dim)

# --- Small MLP block ---
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, act=nn.SiLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act(),
            nn.Linear(hidden, hidden),
            act(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

# --- Denoiser: predicts epsilon(z_t, c, t) ---
class DiffusionDenoiser(nn.Module):
    """
    Input: z_t (B, latent_size), condition c (B, cond_dim), timestep t (B,)
    Output: eps_hat (B, latent_size)
    """
    def __init__(self, latent_size: int, cond_dim: int, time_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.z_proj   = nn.Linear(latent_size, hidden)
        self.c_proj   = nn.Linear(cond_dim, hidden)
        self.t_proj   = nn.Linear(time_dim, hidden)
        self.mlp      = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_size),
        )

    def forward(self, z_t, c, t):
        t_emb = self.time_mlp(t)           # (B, time_dim)
        z_h   = self.z_proj(z_t)           # (B, H)
        c_h   = self.c_proj(c)             # (B, H)
        t_h   = self.t_proj(t_emb)         # (B, H)
        h     = torch.cat([z_h, c_h, t_h], dim=1)
        return self.mlp(h)                 # (B, latent_size)

# --- Standard Gaussian Diffusion (epsilon-prediction) ---
class GaussianDiffusion1D(nn.Module):
    """
    Implements forward noising q(z_t|z_0) and reverse DDPM sampling with an epsilon denoiser.
    """
    def __init__(self, denoiser: DiffusionDenoiser, latent_size: int, timesteps: int = 1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.denoiser   = denoiser
        self.latent_size = latent_size
        self.timesteps  = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        # register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-12))

    def q_sample(self, z0, t, noise=None):
        """ z_t = sqrt(alpha_bar_t) * z0 + sqrt(1-alpha_bar_t)*eps """
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(1)   # (B,1)
        sqrt_omb = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_ab * z0 + sqrt_omb * noise, noise

    @torch.no_grad()
    def p_sample(self, z_t, c, t):
        """
        One reverse step p(z_{t-1}|z_t) using eps-prediction.
        """
        b = z_t.shape[0]
        eps_theta = self.denoiser(z_t, c, t)                        # (B, D)
        beta_t    = self.betas[t].unsqueeze(1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].unsqueeze(1)
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)

        # predict z0
        z0_hat = (z_t - sqrt_one_minus_ab_t * eps_theta) / (self.sqrt_alphas_cumprod[t].unsqueeze(1) + 1e-12)

        # DDPM mean
        mean = sqrt_recip_alpha_t * (z_t - beta_t / sqrt_one_minus_ab_t * eps_theta)

        if (t > 0).all():
            noise = torch.randn_like(z_t)
        else:
            noise = torch.zeros_like(z_t)

        var = self.posterior_variance[t].unsqueeze(1)
        return mean + torch.sqrt(var) * noise, z0_hat

    @torch.no_grad()
    def sample(self, c, num_steps=None):
        """
        c: (B, cond_dim), returns z0 sampled by reverse process
        """
        device = c.device
        T = num_steps if num_steps is not None else self.timesteps
        # if using fewer steps, subsample timesteps uniformly
        if T < self.timesteps:
            t_seq = torch.linspace(self.timesteps-1, 0, T).long().to(device)
        else:
            t_seq = torch.arange(self.timesteps-1, -1, -1, device=device)

        z_t = torch.randn(c.size(0), self.latent_size, device=device)
        z0_hat = None
        for idx, tt in enumerate(t_seq):
            t_batch = torch.full((c.size(0),), int(tt), device=device, dtype=torch.long)
            z_t, z0_hat = self.p_sample(z_t, c, t_batch)
        return z0_hat if z0_hat is not None else z_t

    def loss(self, z0, c):
        """
        DDPM epsilon-prediction loss (simple L2).
        """
        B = z0.size(0)
        device = z0.device
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        z_t, noise = self.q_sample(z0, t)
        eps_hat = self.denoiser(z_t, c, t)
        return F.mse_loss(eps_hat, noise)

# --- Wrapper: use existing Encoder/Decoder to make a Latent Diffusion model ---
class PoseLatentDiffusion(nn.Module):
    """
    Train diffusion in the latent space of (Encoder, Decoder).
    - encoder(x, c) -> z (we use mu only for stability)
    - diffusion learns to denoise z
    - decoder(z, c) -> future pose frames
    """
    def __init__(
        self,
        frame_size: int,
        latent_size: int,
        num_condition_frames: int,
        num_future_predictions: int,
        # pass custom encoder/decoder if you want; else use lightweight defaults
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        diffusion_steps: int = 1000,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 0.02,
        hidden: int = 512,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        # condition is flattened frames of size (B, frame_size * num_condition_frames)
        self.cond_dim = frame_size * num_condition_frames

        # if no encoder/decoder provided, reuse simple MLPs compatible with your file
        if encoder is None:
            self.encoder = Encoder(
                frame_size, latent_size, hidden,
                num_condition_frames, num_future_predictions
            )
        else:
            self.encoder = encoder

        if decoder is None:
            self.decoder = Decoder(
                frame_size, latent_size, hidden,
                num_condition_frames, num_future_predictions
            )
        else:
            self.decoder = decoder

        self.denoiser = DiffusionDenoiser(latent_size, self.cond_dim, time_dim=128, hidden=hidden)
        self.diffusion = GaussianDiffusion1D(
            self.denoiser, latent_size,
            timesteps=diffusion_steps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end
        )

    def encode_mu(self, x, c):
        # Use only mu (deterministic latent) for diffusion target z0
        z, mu, logvar = self.encoder(x, c)
        return mu

    def training_step(self, x, c, recon_weight=1.0, diff_weight=1.0):
        """
        x: future frames flattened (B, num_future_predictions * frame_size)
        c: condition frames flattened (B, num_condition_frames * frame_size)
        Returns: total_loss, dict
        """
        # 1) Latent target z0
        with torch.no_grad():
            z0 = self.encode_mu(x, c)                   # (B, latent)
        # 2) Diffusion eps-loss in latent space
        diff_loss = self.diffusion.loss(z0, c)
        # 3) Optional: reconstruct x from z0 to stabilize decoder
        x_hat = self.decoder(z0, c)
        recon_loss = F.mse_loss(x_hat, x)

        total = recon_weight * recon_loss + diff_weight * diff_loss
        logs = {
            'recon_loss': float(recon_loss.item()),
            'diff_loss': float(diff_loss.item()),
            'total_loss': float(total.item()),
        }
        return total, logs

    @torch.no_grad()
    def sample(self, c, num_steps: int = None, deterministic: bool = False):
        """
        c: (B, cond_dim). Returns predicted future frames (B, num_future_predictions*frame_size)
        """
        # 1) Sample z from diffusion prior conditioned on c
        z = self.diffusion.sample(c, num_steps=num_steps)
        # 2) Decode to future pose frames
        x_hat = self.decoder(z, c)
        return x_hat

    @torch.no_grad()
    def refine(self, x_init, c, num_steps: int = 50, noise_scale: float = 0.1):
        """
        Optional helper: start from encoder's latent and run a few reverse steps for refinement.
        """
        device = x_init.device
        z0 = self.encode_mu(x_init, c)
        z_t = z0 + noise_scale * torch.randn_like(z0)
        # choose a small reverse chain
        t_seq = torch.linspace(self.diffusion.timesteps-1, 0, num_steps).long().to(device)
        z_est = z_t
        for tt in t_seq:
            t_batch = torch.full((c.size(0),), int(tt), device=device, dtype=torch.long)
            z_est, _ = self.diffusion.p_sample(z_est, c, t_batch)
        return self.decoder(z_est, c)
