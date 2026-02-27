"""Perceiver-style transcriptomics autoencoder with ZINB reconstruction,
perceptual, and adversarial losses.

Architecture:
  Encoder: gene input projection -> SetBlock (G->K) -> concat condition tokens
           -> N x Block self-attention -> z (B, K, D)
  Decoder: SetBlock (K->G) -> Linear -> ZINB params (log_mu, log_theta, logit_pi)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

from hooke_forge.model.layers import SetBlock, Block, TransformerCore
from hooke_forge.model.context_encoders import LabelEmbedder

class GeneInputProjection(nn.Module):
    """Embeds each gene as a D-dim token from its scalar count and a learned
    identity embedding.  The count is log1p-transformed before concatenation
    for numerical stability."""

    def __init__(self, n_genes: int, n_embd: int):
        super().__init__()
        self.gene_embeddings = nn.Embedding(n_genes, n_embd)
        self.proj = nn.Linear(n_embd + 1, n_embd)
        self.n_genes = n_genes

    def forward(
        self,
        counts: torch.Tensor,
        gene_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            counts: (B, G) or (B, G') raw/sampled counts.
            gene_indices: optional (G',) long tensor for subsampling genes.
        Returns:
            (B, G, D) gene token embeddings.
        """
        if gene_indices is not None:
            emb = self.gene_embeddings(gene_indices)  # (G', D)
        else:
            emb = self.gene_embeddings.weight  # (G, D)

        emb = emb.unsqueeze(0).expand(counts.shape[0], -1, -1)
        counts_feat = torch.log1p(counts).unsqueeze(-1)  # (B, G, 1)
        x = torch.cat([counts_feat, emb], dim=-1)  # (B, G, D+1)
        return self.proj(x)


def zinb_nll(
    log_mu: torch.Tensor,
    log_theta: torch.Tensor,
    logit_pi: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Zero-inflated negative binomial negative log-likelihood.

    Args:
        log_mu:    (B, G) log of NB mean.
        log_theta: (B, G) log of NB dispersion.
        logit_pi:  (B, G) logit of zero-inflation probability.
        x:         (B, G) raw integer counts.
    Returns:
        Scalar mean NLL.
    """
    eps = 1e-8
    mu = torch.exp(log_mu).clamp(min=eps)
    theta = torch.exp(log_theta).clamp(min=eps)

    # NB log probability using mean-dispersion parameterisation
    log_theta_over_theta_plus_mu = torch.log(theta / (theta + mu))
    log_mu_over_theta_plus_mu = torch.log(mu / (theta + mu))

    log_nb = (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
        + theta * log_theta_over_theta_plus_mu
        + x * log_mu_over_theta_plus_mu
    )

    # Numerically stable ZINB log probability
    log_pi = F.logsigmoid(logit_pi)
    log_one_minus_pi = F.logsigmoid(-logit_pi)

    # NB(0 | mu, theta) shares the theta * log(theta/(theta+mu)) term
    log_nb_zero = theta * log_theta_over_theta_plus_mu

    # x == 0: log(pi + (1-pi)*NB(0)) via logsumexp
    zero_case = torch.logsumexp(
        torch.stack([log_pi, log_one_minus_pi + log_nb_zero], dim=-1),
        dim=-1,
    )
    # x > 0: log(1-pi) + log NB(x)
    nonzero_case = log_one_minus_pi + log_nb

    log_prob = torch.where(x < 0.5, zero_case, nonzero_case)
    return -log_prob.mean()

def sample_zinb(
    log_mu: torch.Tensor,
    log_theta: torch.Tensor,
    logit_pi: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """Differentiable sampling from a ZINB distribution.

    Uses Gamma reparameterisation for the NB component (continuous relaxation
    of the Gamma-Poisson mixture) and Gumbel-Sigmoid for the zero-inflation
    mask.

    Args:
        log_mu:    (B, G) log of NB mean.
        log_theta: (B, G) log of NB dispersion.
        logit_pi:  (B, G) logit of zero-inflation probability.
        tau:       Gumbel-Sigmoid temperature (lower = more discrete-like).
    Returns:
        (B, G) differentiable ZINB samples.
    """
    eps = 1e-8
    mu = torch.exp(log_mu).clamp(min=eps)
    theta = torch.exp(log_theta).clamp(min=eps)

    # Gamma(concentration=theta, rate=theta/mu) has E[rate] = mu
    rate = Gamma(concentration=theta, rate=theta / mu).rsample()

    # Gumbel-Sigmoid relaxation of Bernoulli(1 - pi)
    u = torch.rand_like(logit_pi).clamp(eps, 1 - eps)
    gumbel_noise = torch.log(u) - torch.log(1.0 - u)
    mask = torch.sigmoid((-logit_pi + gumbel_noise) / tau)

    return rate * mask

class TxPerceiverAE(nn.Module):
    """Perceiver-style autoencoder for transcriptomics data.

    Encoder: gene input projection -> SetBlock (G -> K slots) ->
             concat cell_type / assay_type tokens -> N x Block self-attn ->
             z (B, K, D).
    Decoder: SetBlock (K -> G gene queries) -> Linear(D, 3) -> ZINB params.
    """

    def __init__(
        self,
        n_genes: int,
        n_slots: int = 64,
        n_embd: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.0,
        bias: bool = True,
        cell_type_vocab: int = 55,
        assay_type_vocab: int = 6,
        label_dropout_prob: float = 0.15,
        temp_q_init_scale: float = 0.02,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_slots = n_slots
        self.n_embd = n_embd

        self.gene_input = GeneInputProjection(n_genes, n_embd)

        # Encoder: compress G gene tokens -> K slot tokens
        self.encoder_set_block = SetBlock(
            n_template=n_slots,
            n_head=n_heads,
            n_embd=n_embd,
            bias=bias,
            dropout=dropout,
            temp_q_init_scale=temp_q_init_scale,
        )

        # Conditioning (label dropout for classifier-free behaviour)
        self.cell_type_embedder = LabelEmbedder(
            num_classes=cell_type_vocab,
            hidden_size=n_embd,
            dropout_prob=label_dropout_prob,
        )
        self.assay_type_embedder = LabelEmbedder(
            num_classes=assay_type_vocab,
            hidden_size=n_embd,
            dropout_prob=0.0,
        )

        # Self-attention layers on (K+2) tokens
        self.layers = nn.ModuleList([
            Block(
                n_embd=n_embd,
                bias=bias,
                is_causal=False,
                dropout=dropout,
                n_head=n_heads,
            )
            for _ in range(n_layers)
        ])

        # Decoder: expand K slots -> G gene tokens
        self.decoder_set_block = SetBlock(
            n_template=n_genes,
            n_head=n_heads,
            n_embd=n_embd,
            bias=bias,
            dropout=dropout,
            temp_q_init_scale=temp_q_init_scale,
        )

        # ZINB output heads: log_mu, log_theta, logit_pi
        self.output_proj = nn.Linear(n_embd, 3)

        # Latent normalisation buffers (populated offline before diffusion)
        self.register_buffer("latent_mean", torch.zeros(1, 1, n_embd))
        self.register_buffer("latent_std", torch.ones(1, 1, n_embd))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        n_res = len(self.layers)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * n_res)
                )

    def encode(
        self,
        raw_counts: torch.Tensor,
        cell_type: torch.Tensor,
        assay_type: torch.Tensor,
        train: bool = True,
    ) -> torch.Tensor:
        """Encode raw counts to latent z (B, K, D)."""
        gene_tokens = self.gene_input(raw_counts)  # (B, G, D)
        slots = self.encoder_set_block(gene_tokens)  # (B, K, D)

        cell_emb = self.cell_type_embedder(cell_type, train=train)  # (B, D)
        assay_emb = self.assay_type_embedder(assay_type, train=train)  # (B, D)

        x = torch.cat([
            slots,
            cell_emb.unsqueeze(1),
            assay_emb.unsqueeze(1),
        ], dim=1)  # (B, K+2, D)

        for block in self.layers:
            x = block(x)

        return x[:, : self.n_slots]  # (B, K, D)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode latent z to ZINB parameters."""
        gene_tokens = self.decoder_set_block(z)  # (B, G, D)
        params = self.output_proj(gene_tokens)  # (B, G, 3)
        return {
            "log_mu": params[..., 0],
            "log_theta": params[..., 1],
            "logit_pi": params[..., 2],
        }

    def forward(
        self,
        raw_counts: torch.Tensor,
        cell_type: torch.Tensor,
        assay_type: torch.Tensor,
        train: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encode(raw_counts, cell_type, assay_type, train=train)
        zinb_params = self.decode(z)
        return z, zinb_params

    # ---- Latent normalisation helpers (for downstream diffusion) -------------

    def encode_normalised(self, raw_counts, cell_type, assay_type, train=False):
        z = self.encode(raw_counts, cell_type, assay_type, train=train)
        return (z - self.latent_mean) / self.latent_std.clamp(min=1e-6)

    def decode_denormalised(self, z_norm):
        z = z_norm * self.latent_std.clamp(min=1e-6) + self.latent_mean
        return self.decode(z)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


class TxDiscriminator(nn.Module):
    """Gene-sampling Transformer discriminator.

    Each forward pass randomly sub-samples ``n_disc_genes`` genes and feeds
    them through a Transformer, making the architecture invariant to the total
    gene count.  The gene identity embeddings + input projection are shared
    with the autoencoder (passed at forward time, not owned by this module).
    """

    def __init__(
        self,
        n_embd: int = 512,
        n_disc_genes: int = 1000,
        n_layer: int = 3,
        n_head: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.n_disc_genes = n_disc_genes
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_embd) * 0.02)
        self.transformer = TransformerCore(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
            is_causal=False,
        )
        self.head = nn.Linear(n_embd, 1)

    def forward(
        self,
        counts: torch.Tensor,
        gene_input_proj: GeneInputProjection,
    ) -> torch.Tensor:
        """
        Args:
            counts: (B, G) count values (raw counts or ZINB samples).
            gene_input_proj: shared GeneInputProjection from the AE.
        Returns:
            (B, 1) real/fake logits.
        """
        B, G = counts.shape
        n_sample = min(self.n_disc_genes, G)

        idx = torch.randperm(G, device=counts.device)[:n_sample].sort().values
        sub_counts = counts[:, idx]

        tokens = gene_input_proj(sub_counts, idx)  # (B, n_sample, D)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        tokens = self.transformer(tokens)  # (B, 1+n_sample, D)
        return self.head(tokens[:, 0])  # (B, 1)


def hinge_disc_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """Hinge loss for the discriminator (VQGAN / LDM style)."""
    return (F.relu(1.0 - real_logits) + F.relu(1.0 + fake_logits)).mean()


# ---------------------------------------------------------------------------
# TxAM Perceptual Loss
# ---------------------------------------------------------------------------

class TxAMPerceptualLoss(nn.Module):
    """TxAM-based perceptual loss for transcriptomics data.

    Uses a pre-trained TxAM encoder to extract embeddings from transcriptomics
    data and computes perceptual loss in the TxAM embedding space.
    """

    def __init__(
        self,
        checkpoint_path: str = "/rxrx/data/valence/hooke/predict/txam_checkpoints/TxAM_TREK_v1/checkpoint.pt",
        device: str = "cuda",
        input_gene_names: list[str] | None = None,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device

        try:
            # Import TxAM encoder
            from txam import TxAMEncoder

            # Load pre-trained TxAM encoder wrapper
            txam_encoder = TxAMEncoder.from_pretrained(
                checkpoint_path=checkpoint_path,
                device=device
            )

            # Get the raw PyTorch encoder module for gradient-enabled inference
            self.encoder = txam_encoder.get_encoder_module()
            self.preprocessor = txam_encoder.preprocessor

            # Set encoder to evaluation mode and freeze parameters
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Cache preprocessing parameters
            self.target_library_size = getattr(
                self.preprocessor, 'target_library_size', 10_000.0
            )

        except ImportError as e:
            raise ImportError(
                f"Failed to import txam: {e}. "
                "Please ensure txam package is available."
            ) from e

        # --- Gene alignment (one-time) ---
        if input_gene_names is not None:
            model_genes = self.preprocessor.gene_names
            model_gene_to_idx = {g: i for i, g in enumerate(input_gene_names)}
            # For each model gene, find its column in the input (or -1 if missing)
            idx = []
            for g in model_genes:
                idx.append(model_gene_to_idx.get(g, -1))
            self.register_buffer(
                '_align_idx', torch.tensor(idx, dtype=torch.long)
            )
        # If input_gene_names is None, no _align_idx buffer is created

    def _preprocess_counts(self, counts: torch.Tensor) -> torch.Tensor:
        """Apply TxAM preprocessing: normalize + log1p transform.

        Args:
            counts: (B, G) raw counts tensor

        Returns:
            (B, G) preprocessed tensor
        """
        # Normalize per cell to target library size
        counts_sum = counts.sum(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = (counts / counts_sum) * self.target_library_size

        # Apply log1p transform
        return torch.log1p(normalized)

    def _align(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder/pad input columns to match model gene order."""
        if not hasattr(self, '_align_idx'):
            return x
        # Pad a zero column at the end for missing genes (index == -1)
        padded = F.pad(x, (0, 1), value=0.0)          # (B, G_input+1)
        idx = self._align_idx.clamp(min=-1) % padded.shape[1]
        # gather is not needed; advanced indexing works and is differentiable
        return padded[:, idx]                           # (B, G_model)

    def forward(
        self,
        recon: torch.Tensor,
        real: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss between reconstructed and real transcriptomics data.

        Args:
            recon: (B, G) ZINB samples (differentiable).
            real:  (B, G) raw counts.

        Returns:
            Scalar perceptual loss in TxAM embedding space.
        """
        # Align gene order to match model expectations
        recon = self._align(recon)
        real = self._align(real)

        # Preprocess both inputs
        recon_preprocessed = self._preprocess_counts(recon)

        with torch.no_grad():
            real_preprocessed = self._preprocess_counts(real)

        # Extract embeddings using the raw encoder (preserves gradients for recon)
        recon_embeddings = self.encoder(recon_preprocessed)

        with torch.no_grad():
            real_embeddings = self.encoder(real_preprocessed)

        # Compute MSE loss between embeddings
        return F.mse_loss(recon_embeddings, real_embeddings)



def test_txam_gradient_flow(
    checkpoint_path: str = "/rxrx/data/valence/hooke/predict/txam_checkpoints/TxAM_TREK_v1/checkpoint.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """Test that gradients flow correctly through TxAMPerceptualLoss.

    Args:
        checkpoint_path: Path to TxAM checkpoint
        device: Device to run test on
    """
    print("Testing TxAM perceptual loss gradient flow...")

    try:
        # Create test loss function
        loss_fn = TxAMPerceptualLoss(checkpoint_path=checkpoint_path, device=device)

        # Create dummy data with realistic transcriptomics values
        n_genes = loss_fn.preprocessor.num_genes
        batch_size = 4

        # Real data (no gradients needed)
        real_counts = torch.poisson(torch.ones(batch_size, n_genes, device=device) * 5.0)

        # Reconstructed data (needs gradients)
        recon_counts = torch.poisson(torch.ones(batch_size, n_genes, device=device) * 5.0)
        recon_counts.requires_grad_(True)

        print(f"Input shape: {recon_counts.shape}")
        print(f"Target library size: {loss_fn.target_library_size}")

        # Compute loss
        loss = loss_fn(recon_counts, real_counts)

        print(f"Loss value: {loss.item():.4f}")

        # Test gradient computation
        loss.backward()

        # Check if gradients were computed
        if recon_counts.grad is not None:
            grad_norm = recon_counts.grad.norm().item()
            grad_mean = recon_counts.grad.mean().item()
            print(f"Gradient norm: {grad_norm:.4f}")
            print(f"Gradient mean: {grad_mean:.6f}")
            print("✓ Gradients computed successfully!")
            print("✓ TxAM perceptual loss is working correctly!")
        else:
            print("✗ No gradients computed - gradient flow is broken!")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run test if script is executed directly
    test_txam_gradient_flow()
