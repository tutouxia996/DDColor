import cv2
import numpy as np
import torch
import torch.nn.functional as F


def load_checkpoint_state_dict(model_path: str, map_location="cpu"):
    """Load a checkpoint and return a state_dict.

    Supports both:
    - {'params': state_dict, ...} (common in this repo)
    - raw state_dict
    """
    ckpt = torch.load(model_path, map_location=map_location)
    if isinstance(ckpt, dict) and "params" in ckpt:
        return ckpt["params"]
    return ckpt


def build_ddcolor_model(
    model_cls,
    *,
    model_path: str,
    input_size: int = 512,
    model_size: str = "large",
    decoder_type: str = "MultiScaleColorDecoder",
    device=None,
    **kwargs,
):
    """Build a DDColor model and load weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_size not in ("tiny", "large"):
        raise ValueError(f"model_size must be 'tiny' or 'large', got: {model_size}")
    encoder_name = "convnext-t" if model_size == "tiny" else "convnext-l"

    if decoder_type == "MultiScaleColorDecoder":
        kwargs.setdefault("num_queries", 100)
        kwargs.setdefault("num_scales", 3)
        kwargs.setdefault("dec_layers", 9)
    elif decoder_type == "SingleColorDecoder":
        kwargs.setdefault("num_queries", 256)
    else:
        raise NotImplementedError(f"decoder_type not implemented: {decoder_type}")

    model = model_cls(
        encoder_name=encoder_name,
        decoder_name=decoder_type,
        input_size=[input_size, input_size],
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        **kwargs,
    )

    state_dict = load_checkpoint_state_dict(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def _guided_upsample_ab(ab_low: np.ndarray, guide_l: np.ndarray, out_hw) -> np.ndarray:
    """Upsample ab with L-channel guidance to reduce red/blue edge fringing.

    ab_low: (h, w, 2) float
    guide_l: (H, W) float luminance in [0,1] or [0,100]
    """
    H, W = out_hw
    a = cv2.resize(ab_low[:, :, 0], (W, H), interpolation=cv2.INTER_CUBIC)
    b = cv2.resize(ab_low[:, :, 1], (W, H), interpolation=cv2.INTER_CUBIC)

    # Normalize guide to 0-255 for edge detection
    g = guide_l.astype(np.float32)
    g = g - g.min()
    g = g / (g.max() + 1e-6)
    g8 = (g * 255.0).astype(np.uint8)

    # Bilateral on chroma (edge-preserving) kills zipper / red-blue fringes
    a_s = cv2.bilateralFilter(a.astype(np.float32), d=7, sigmaColor=0.05, sigmaSpace=7)
    b_s = cv2.bilateralFilter(b.astype(np.float32), d=7, sigmaColor=0.05, sigmaSpace=7)

    edges = cv2.Canny(g8, 60, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (7, 7), 0)
    edges = edges[..., None]

    a = a * (1.0 - edges[..., 0] * 0.85) + a_s * (edges[..., 0] * 0.85)
    b = b * (1.0 - edges[..., 0] * 0.85) + b_s * (edges[..., 0] * 0.85)
    return np.stack([a, b], axis=-1)


class ColorizationPipeline:
    """Shared image colorization pipeline used by CLI/Gradio/Cog.

    - input: BGR uint8 image (OpenCV)
    - output: BGR uint8 image (OpenCV)
    """

    def __init__(self, model, *, input_size: int = 512, device=None):
        self.input_size = int(input_size)
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            if img_bgr is None:
                raise ValueError("img is None (cv2.imread failed?)")

            height, width = img_bgr.shape[:2]

            # True neutral gray RGB (avoid yellowed paper confusing the net)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            img = (gray_bgr / 255.0).astype(np.float32)
            orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # float Lab L

            img_resized = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            # Feed proper grayscale RGB, not Lab-with-zero-ab roundtrip quirks
            gray_rgb = cv2.cvtColor(
                cv2.resize(gray, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA),
                cv2.COLOR_GRAY2RGB,
            ).astype(np.float32) / 255.0

            tensor_gray_rgb = (
                torch.from_numpy(gray_rgb.transpose((2, 0, 1)))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

            output_ab = self.model(tensor_gray_rgb).cpu()  # (1, 2, S, S)
            ab_low = output_ab[0].float().numpy().transpose(1, 2, 0)  # (S,S,2)

            output_ab_resized = _guided_upsample_ab(ab_low, orig_l[:, :, 0], (height, width))
            output_lab = np.concatenate((orig_l, output_ab_resized), axis=-1)
            output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
            output_img = (np.clip(output_bgr, 0, 1) * 255.0).round().astype(np.uint8)
            return output_img
