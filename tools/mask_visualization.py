import os
import torch
from torchvision.utils import make_grid, save_image

def visualize_masked_patches(
    img: torch.Tensor,
    patch_positions: list,
    save_dir: str = 'mask_visualizations',
    image_size: int = 336,
    patch_size: int = 14,
    alpha: float = 0.6
) -> None:
    """Generate side-by-side comparison of original and masked image."""
    
    # Handle single image input
    images = img.unsqueeze(0) if img.dim() == 3 else img
    
    # Core parameters
    num_patches = image_size // patch_size
    masked_img = images.clone()
    mask = torch.ones(num_patches, num_patches, dtype=torch.bool, device=img.device)

    # Create mask from positions
    for pos in map(int, patch_positions):
        row, col = divmod(pos, num_patches)
        mask[row, col] = False

    # Apply masks vectorized
    for r, c in torch.stack(torch.where(mask)).t():
        y, x = r*patch_size, c*patch_size
        masked_img[0, :, y:y+patch_size, x:x+patch_size] = \
            (1-alpha)*masked_img[0, :, y:y+patch_size, x:x+patch_size] + alpha

    # Generate and save comparison
    os.makedirs(save_dir, exist_ok=True)
    grid = make_grid(
        torch.cat([images, masked_img], 0),
        nrow=1,
        padding=2,
        normalize=True
    )
    save_image(grid, os.path.join(save_dir, 'mask_comparison.png'))
