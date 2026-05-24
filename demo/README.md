# Pix2Pix Facades Demo

This repository contains an inference demo for an image-to-image translation model trained on the Facades dataset.

The model receives a semantic label map of a building facade and generates a realistic facade image using a trained Pix2Pix generator.

## Project description

The objective of this project is to use image-to-image translation models, such as Pix2Pix, to generate realistic images from input label maps.

In this demo, the input is a facade label map and the output is a generated realistic facade image.

## Installation

Run the following commands from the folder that contains `inference.py`.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running inference

To generate a facade image from the sample label map, run:

```bash
python inference.py --input samples/sample_label_map.jpg --model models/pix2pix_generator.pt --output outputs/generated_sample.png
```

The generated image will be saved in:

```text
outputs/generated_sample.png
```

## Input format

The input image must be an RGB semantic label map.

The image is automatically resized to 256 x 256 before being passed to the generator.

If the input image is a paired Facades image, where the real facade is on the left and the label map is on the right, run:

```bash
python inference.py --input samples/paired_image.jpg --model models/pix2pix_generator.pt --output outputs/generated_sample.png --paired
```

In that case, the script automatically crops the right half of the image and uses it as the label map.

## Model

The model used in this demo is an exported trained Pix2Pix generator.

The generator receives a label map and produces a realistic facade image.

The model was exported as a TorchScript file so that the inference script can load it directly without redefining the full architecture.

## Files

### `inference.py`

Python script used to run inference with the trained generator.

### `requirements.txt`

List of Python dependencies required to run the demo.

### `models/pix2pix_generator.pt`

Exported trained generator model.

### `samples/sample_label_map.jpg`

Example input label map.

### `outputs/`

Folder where generated images are saved.

## Notes

This repository only contains the inference demo.
