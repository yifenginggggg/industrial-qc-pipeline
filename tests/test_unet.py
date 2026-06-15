import torch

from industrial_qc.models.unet import UNet


def test_unet_output_matches_input_spatial_shape():
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 128, 128)
    y = model(x)

    assert tuple(y.shape) == (2, 1, 128, 128)
