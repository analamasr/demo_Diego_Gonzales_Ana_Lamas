# Pix2Pix Facades Demo

This repository contains an inference demo for an image-to-image translation model trained on the Facades dataset.

The model receives a semantic label map of a building facade and generates a realistic facade image using a trained Pix2Pix generator.

## Project description

The objective of this project is to use image-to-image translation models, such as Pix2Pix, to generate realistic images from input label maps.

In this demo, the input is a facade label map and the output is a generated realistic facade image.

## Repository structure

```text
demo_cv2_proyect1/
├── task3_demo.ipynb
├── pix2pix_generator.pt        # downloaded separately
└── demo/
    ├── inference.py
    ├── requirements.txt
    ├── samples/
    │   └── sample_label_map.jpg
    └── outputs/
```

## Downloading the trained model

The trained model is not included in this GitHub repository because of file size limitations.

Before running the demo, download the exported TorchScript model from the following OneDrive link:

```text
https://nubeusc-my.sharepoint.com/:u:/g/personal/ana_lamas_rodriguez_rai_usc_es/IQB8AYkMn1xJRbuzNdHrpJy4AUcmkkpZEIfU5Eg4TFni4fI?e=G2w4VL
```

After downloading it, place the model file in the root folder of this repository, at the same level as `task3_demo.ipynb`.

The expected location is:

```text
demo_cv2_proyect1/pix2pix_generator.pt
```

## Installation

From the root folder of the repository, create and activate a virtual environment:

```bash
python -m venv demo_env
source demo_env/bin/activate
```

On Windows, activate the environment with:

```bash
demo_env\Scripts\activate
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running inference from terminal

To generate a facade image from the sample label map, run the following command from the root folder of the repository:

```bash
python demo/inference.py --input demo/samples/sample_label_map.jpg --model pix2pix_generator.pt --output demo/outputs/generated_sample.png
```

The generated image will be saved in:

```text
demo/outputs/generated_sample.png
```

## Running inference with a paired Facades image

If the input image is a paired Facades image, where the real facade is on the left and the label map is on the right, run:

```bash
python demo/inference.py --input demo/samples/paired_image.jpg --model pix2pix_generator.pt --output demo/outputs/generated_sample.png --paired
```

In that case, the script automatically crops the right half of the image and uses it as the label map.

## Running the notebook

The demo can also be executed directly from the notebook:

```text
task3_demo.ipynb
```

Before running the notebook, download `pix2pix_generator.pt` from the OneDrive link above and place it in the root folder of the repository, at the same level as the notebook.

Then open the notebook in Jupyter Notebook, JupyterLab or VS Code and run the cells in order.

If you are using a virtual environment and need to select it as a notebook kernel, run:

```bash
python -m ipykernel install --user --name demo_env --display-name "demo_env"
```

Then select the `demo_env` kernel in the notebook.

## Input format

The input image must be an RGB semantic label map.

The image is automatically resized to 256 x 256 before being passed to the generator.

## Model

The model used in this demo is the selected M4 Pix2Pix generator, exported as a TorchScript file.

M4 was selected as the final model because it obtained the best overall performance in the project evaluation. It combines resize-convolution decoding, edge loss, LSGAN, spectral normalization and perceptual loss.

The TorchScript export allows the inference script to load the trained generator directly without redefining the full architecture.

## Files

### `task3_demo.ipynb`

Notebook version of the demo.

### `demo/inference.py`

Python script used to run inference with the trained generator.

### `demo/requirements.txt`

List of Python dependencies required to run the demo.

### `pix2pix_generator.pt`

Exported trained generator model. This file must be downloaded separately from the OneDrive link and placed in the root folder of the repository.

### `demo/samples/sample_label_map.jpg`

Example input label map.

### `demo/outputs/`

Folder where generated images are saved. The folder is created automatically if it does not exist.

## Notes

This repository only contains the inference demo. It does not include the full training pipeline.

The trained model must be downloaded separately before running inference.
