import torch
import diffusers

class StabilityCPEncoder:
    def __init__(self, device: torch.device, compile: bool = True):
        self.vae = diffusers.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")  # type: ignore
        self.vae.eval().requires_grad_(False).to(device)  # type: ignore
        self.scale = 0.18215  # magic constant from the DiT codebase

        if compile:
            self.encode = torch.compile(self.encode)
            self.decode = torch.compile(self.decode)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of cell-paint images into the stable diffusion latent space.
        input: torch.uint8 [0, 255], (B, 6, H, W)
        output: torch.float32, (B, 8, H // 8, W // 8)"""
        x = x.to(torch.float32) / 127.5 - 1
        x0, x1 = torch.chunk(x, 2, dim=1)
        x0 = self.vae.encode(x0).latent_dist.sample()  # type: ignore
        x1 = self.vae.encode(x1).latent_dist.sample()  # type: ignore
        return torch.cat([x0, x1], dim=1) * self.scale

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of stable diffusion latents into cell-paint space.
        input: torch.float32, (B, 8, H // 8, W // 8)
        output: torch.uint8 [0, 255], (B, 6, H, W)"""
        x = x / self.scale
        x0, x1 = torch.chunk(x, 2, dim=1)
        x0 = self.vae.decode(x0).sample  # type: ignore
        x1 = self.vae.decode(x1).sample  # type: ignore
        x = torch.cat([x0, x1], dim=1)
        x = (x + 1) * 127.5
        x = x.clip(0, 255).to(torch.uint8)
        return x

class DINOv2Detector:
    def __init__(self, device: torch.device):
        self.model = torch.hub.load(
            "facebookresearch/dinov2:main",
            "dinov2_vitl14",
            trust_repo=True,
            verbose=False,
            skip_validation=True,
        )
        assert isinstance(self.model, torch.nn.Module)
        self.model.eval().requires_grad_(False).to(device)

        self.MEAN = torch.tensor(
            [0.485, 0.456, 0.406], device=device, dtype=torch.float32
        ).view(1, -1, 1, 1)
        self.STD = torch.tensor(
            [0.229, 0.224, 0.225], device=device, dtype=torch.float32
        ).view(1, -1, 1, 1)

    def __call__(self, x):
        """Extract features using DinoV2.
        Input is expected to be a torch.uint8 tensor of shape (B, 3, H, W)."""
        x = torch.nn.functional.interpolate(
            x.to(torch.float32),
            size=(224, 224),
            mode="bicubic",
            antialias=True,
        )
        x = x.to(torch.float32) / 255  # to float32 in [0,1]
        x = (x - self.MEAN) / self.STD
        return self.model(x)  # type: ignore


def rewire_graph_to_return_z(model_to_modify: torch.jit.ScriptModule):
    """
    Modifies the graph to return only the intermediate tensor 'z'
    instead of the final output.
    """
    graph = model_to_modify.forward.graph
    mean_node = None
    z_tensor_node = None
    for node in reversed(list(graph.nodes())):
        if node.kind() == "aten::mean":
            mean_node = node
            z_tensor_node = next(mean_node.inputs())
            break
    if not mean_node:
        print("Error: Could not find an 'aten::mean' node.")
        return None
    num_outputs = len(list(graph.outputs()))
    for _ in range(num_outputs):
        graph.eraseOutput(0)
    graph.registerOutput(z_tensor_node)
    return model_to_modify

class Phenom2Detector:
    def __init__(self, device: torch.device, rewire: bool = False):
        self.device = device
        self.model = torch.jit.load(
            "/mnt/ps/home/CORP/jason.hartford/project/cj-generative/artifacts/inference_model/model.pth",
            map_location=device,
        )
        if rewire:
            self.model = rewire_graph_to_return_z(self.model)
        self.model.eval().to(device)

    def __call__(self, x):
        """Extract features using Phenom-2.
        Input is expected to be a torch.uint8 tensor of shape (B, 6, 256, 256)."""
        x = x.to(self.device, non_blocking=True)
        return self.model(x)


class PH2BFDetector:
    def __init__(self, device: torch.device, rewire: bool = False):
        from pathlib import Path

        self.device = device
        model_path = Path(
            "/rxrx/data/valence/hooke-models/PH2-BF/inference_model/model.pth"
        )
        self.model = torch.jit.load(str(model_path), map_location=device)
        if rewire:
            self.model = rewire_graph_to_return_z(self.model)
        self.model.eval().to(device)

    def __call__(self, x):
        """Extract features using PH2-BF.
        Input is expected to be a torch.uint8 tensor of shape (B, 3, 256, 256)."""
        B, c, h, w = x.shape
        if c != 3:
            x = x[:, :3, :, :]
        x = x.to(self.device, non_blocking=True)
        return self.model(x)
