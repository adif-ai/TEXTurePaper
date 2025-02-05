from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger


@dataclass
class RenderConfig:
    """ Parameters for the Mesh Renderer """
    # Grid size for rendering during painting
    train_grid_size: int = 2048
    # Grid size of evaluation
    eval_grid_size: int = 1024
    # training camera radius range
    radius: float = 1.5
    # Set [0,overhead_range] as the overhead region
    overhead_range: float = 40
    # Define the front angle region
    front_range: float = 70
    # The front offset, use to rotate shape from code
    front_offset:float = 0.0
    # Number of views to use
    n_views: int = 8
    # Theta value for rendering during training
    base_theta:float = 60
    # Additional views to use before rotating around shape
    views_before: List[Tuple[float,float]] = field(default_factory=[[180,1],[180,179]].copy)
    # Additional views to use after rotating around shape
    views_after: List[Tuple[float, float]] = field(default_factory=[[0,30], [180,150], [0,150]].copy)
    # Whether to alternate between the rotating views from the different sides
    alternate_views: bool = True

@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    # Guiding text prompt
    text: str = ""
    # Guiding added text prompt
    added_text: str = "{} view, consistent, best quality, extremely detailed, 8k, raw photo, highres, realistic, ultra detailed"
    # Guiding negative text prompt
    negative_text: str = ""
    added_negative_text: str = "cracks, scratches, shadows, blobs, stain, light reflections, reflections, light, lowres, extra digit, fewer digits, cropped, worst quality, low quality"
    # The mesh to paint
    shape_path: str = None
    # Reference image Path
    reference_image_path: str = None
    reference_image_repeat: int = 1
    # Refererence style fidelity
    style_fidelity: float = 0.5
    # diffusion model to use
    diffusion_name: str = "v1-5-pruned-emaonly.safetensors"
    # Scale of mesh in 1x1x1 cube
    shape_scale: float = 0.6
    # height of mesh
    dy: float = 0.25
    # texture image resolution
    texture_resolution: int = 2048
    # texture mapping interpolation mode from texture image, options: 'nearest', 'bilinear', 'bicubic'
    texture_interpolation_mode: str= 'bilinear'
    # The texture before editing
    reference_texture: Optional[Path] = None
    # The edited texture
    initial_texture: Optional[Path] = None
    # Whether to use background color or image
    use_background_color: bool = False
    # Background image to use
    background_img: str = 'textures/white_room.png'
    # Threshold for defining refine regions
    z_update_thr: float = 0.5
    # Use absolute threshold (use difference between z_normals_cache if False)
    z_update_abs: bool = True
    # Some more strict masking for projecting back
    use_refine: bool = True
    # Mask dilation
    use_dilation: bool = False
    # Checkerboard masking
    use_checkerboard: bool = False
    # Stabld Diffusion resolution
    image_resolution: int = 1024
    # denoising_strength
    denoising_strength: float = 0.75
    # upscale
    upscale: bool = False
    upscale_resize: float = 4
    upscaler1: int = 7


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Seed for experiment
    seed: int = 0
    # Learning rate for projection
    lr: float = 1e-2
    # For Diffusion model
    min_timestep: float = 0.02
    # For Diffusion model
    max_timestep: float = 0.98
    # For Diffusion model
    no_noise: bool = False
    # Diffusion steps
    steps: int = 20


@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Experiment name
    exp_name: str
    # Experiment output dir
    exp_root: Path = Path('experiments/')
    # How many steps between save step
    save_interval: int = 100
    # Run only test
    eval_only: bool = False
    # Number of angles to sample for eval during training
    eval_size: int = 10
    # Number of angles to sample for eval after training
    full_eval_size: int = 100
    # Export a mesh
    save_mesh: bool = True
    # Whether to show intermediate diffusion visualizations
    vis_diffusion_steps: bool = False
    # Whether to log intermediate images
    log_images: bool = True

    @property
    def exp_dir(self) -> Path:
        return self.exp_root / self.exp_name


@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)


