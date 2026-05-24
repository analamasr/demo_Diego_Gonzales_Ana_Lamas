import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResizeConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), dim=1)
        return x


class GeneratorUNetResize(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = ResizeConvUp(512, 512, dropout=0.5)
        self.up2 = ResizeConvUp(1024, 512, dropout=0.5)
        self.up3 = ResizeConvUp(1024, 512, dropout=0.5)
        self.up4 = ResizeConvUp(1024, 512, dropout=0.5)
        self.up5 = ResizeConvUp(1024, 256)
        self.up6 = ResizeConvUp(512, 128)
        self.up7 = ResizeConvUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


def load_input_image(image_path, paired=False):
    image = Image.open(image_path).convert("RGB")

    if paired:
        width, height = image.size
        mid = width // 2
        image = image.crop((mid, 0, width, height))

    transform = transforms.Compose([
        transforms.Resize(
            (256, 256),
            interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    return transform(image).unsqueeze(0)


def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(tensor)


def load_generator(checkpoint_path, device):
    model = GeneratorUNetResize().to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Pix2Pix Facades inference demo using a .pth checkpoint."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input label map image."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="m4_generator_perc_5.pth",
        help="Path to the trained .pth generator checkpoint."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="demo/outputs/generated_image.png",
        help="Path where the generated image will be saved."
    )

    parser.add_argument(
        "--paired",
        action="store_true",
        help=(
            "Use this flag if the input is a paired Facades image. "
            "The right half will be used as the label map."
        )
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_generator(args.model, device)

    input_tensor = load_input_image(args.input, paired=args.paired).to(device)

    with torch.no_grad():
        generated_tensor = model(input_tensor)

    output_image = tensor_to_image(generated_tensor)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output_path)

    print("Generated image saved to:", output_path)


if __name__ == "__main__":
    main()