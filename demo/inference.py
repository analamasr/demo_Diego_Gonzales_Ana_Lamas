
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_input_image(image_path, paired=False):
    image = Image.open(image_path).convert("RGB")

    if paired:
        width, height = image.size
        mid = width // 2
        image = image.crop((mid, 0, width, height))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
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


def main():
    parser = argparse.ArgumentParser(
        description="Pix2Pix Facades inference demo."
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
        default="models/pix2pix_generator.pt",
        help="Path to the exported TorchScript generator model."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generated_image.png",
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

    model = torch.jit.load(args.model, map_location=device)
    model.to(device)
    model.eval()

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
