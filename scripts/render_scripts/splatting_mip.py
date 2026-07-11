#%%
import torch
from dataclasses import dataclass

@dataclass
class camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class Gaussian3D:
    means: torch.Tensor
    scales: torch.Tensor
    quaternions: torch.Tensor
    intensities: torch.Tensor

def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.
    Args:
        quaternions (torch.Tensor): Tensor of shape (N, 4) representing N quaternions.
    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3) representing N rotation matrices.
    """
    # Normalize the quaternions
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

    w, x, y, z = quaternions.unbind(-1)
    
    # Compute the rotation matrix elements
    R = torch.stack([
        1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)
    ], dim=-1).reshape(-1, 3, 3)

    return R
#%%

# test quaternion_to_rotation_matrix
import torch
gaussian_params = torch.load("models_smoke/block_z001_y001_x006/best.pth")
means = gaussian_params['means']
scales_log = gaussian_params['log_scales']
quaternions = gaussian_params['quats']
intensities = gaussian_params['intensities']
print(f'means shape: {means.shape}')
print(f'scales_log shape: {scales_log.shape}')
print(f'quaternions shape: {quaternions.shape}')
print(f'intensities shape: {intensities.shape}')

test_rotation_matrices = quaternion_to_rotation_matrix(quaternions[:1])
# try 1 gaussian to check wether it is according our quats or not
test_gaussian = Gaussian3D(
    means=means[0],
    scales=torch.exp(scales_log[0]),
    quaternions=quaternions[0],
    intensities=intensities[0]
)
# what should the results be? we can check the rotation matrix of the first gaussian
print(f'test_gaussian means: {test_gaussian.means}')
print(f'test_gaussian scales: {test_gaussian.scales}')
print(f'test_gaussian quaternions: {test_gaussian.quaternions}')
print(f'rotation_matrices: {test_rotation_matrices[0]}')
print(f'test_gaussian intensities: {test_gaussian.intensities}')


# %%
