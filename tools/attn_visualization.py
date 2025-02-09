import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_attn_scores(
    attn_scores, 
    output_folder: str = 'attn_visualizations', 
    patch_size: int = 24,
    colormap: str = 'viridis',
    dpi: int = 300
) -> None:
    """Visualize attention scores as a heatmap and save to file."""
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Input validation and tensor preprocessing
    if not isinstance(attn_scores, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
        
    score_map = attn_scores.cpu().detach().numpy().reshape(patch_size, patch_size)

    # Create figure with normalized color mapping
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    plt.imshow(
        score_map,
        cmap=colormap,
        norm=plt.Normalize(np.min(score_map), np.max(score_map)),
        interpolation='nearest'
    )
    plt.axis('off')

    # Save with optimized image parameters
    plt.savefig(
        os.path.join(output_folder, 'attn_score.png'),
        bbox_inches='tight',
        pad_inches=0,
        transparent=True
    )
    plt.close(fig)
