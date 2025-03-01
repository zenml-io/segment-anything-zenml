# SAM Fine-tuning ZenML Pipeline

This repository contains a ZenML pipeline for fine-tuning the Segment Anything Model (SAM) and generating interactive visualizations of segmentation results.

## Overview

The pipeline performs the following steps:
1. Loads images and prepares synthetic masks for training
2. Fine-tunes the SAM model on prepared data
3. Generates segmentation masks and creates visualizations
4. Produces an interactive HTML artifact for comparing results

## Prerequisites

- Python 3.8+
- Mac M1/M2 (Apple Silicon) or system with NVIDIA GPU
- Docker (for containerized execution)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/zenml-io/segment-anything-zenml
   cd sam-zenml-pipeline
   ```

2. Run the setup script to create directories, download sample images, and install dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   The setup script will:
   - Create project directories
   - Download sample images
   - Download the SAM model checkpoint
   - Set up a Python virtual environment
   - Install dependencies
   - Initialize ZenML

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

## Running the Pipeline

### Basic Execution

Run the pipeline with default settings:

```bash
python sam_pipeline.py
```

This will:
- Load images from the `./images` directory
- Use the SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`)
- Save outputs to the `./output` directory
- Run for 2 training epochs

### Custom Configuration

You can modify the main script to customize parameters:

```python
if __name__ == "__main__":
    sam_pipeline(
        image_dir="./custom_images",
        checkpoint_path="./sam_vit_h_4b8939.pth",  # For higher quality
        output_dir="./custom_output",
        epochs=5  # More training epochs
    )
```

## Viewing Results

### Output Files

The pipeline generates several files for each processed image:
- `original_*.jpg`: Original input image
- `mask_*.jpg`: Binary segmentation mask
- `overlay_*.jpg`: Green overlay visualization

### Interactive Visualization

The pipeline creates an interactive HTML visualization that can be viewed in the ZenML dashboard.

1. Start the ZenML UI if not already running:
   ```bash
   zenml up
   ```

2. Open the dashboard (typically at http://127.0.0.1:8237)

3. Navigate to:
   - Pipelines → sam_pipeline → Latest run → Artifacts
   - Find the HTML artifact created by the `create_interactive_html_artifact` step
   - Click to open the interactive visualization

4. Using the visualization:
   - Select different images from the dropdown
   - Compare original images with masks or overlays
   - Toggle between different view types

## Customizing the Pipeline

### Using Your Own Images

Place your images in the `./images` directory or specify a custom directory when running the pipeline.

### Using Different SAM Models

The pipeline supports different SAM model variants:
- ViT-B (default): `sam_vit_b_01ec64.pth` - Balance of speed and quality
- ViT-L: `sam_vit_l_0b3195.pth` - Higher quality, slower
- ViT-H: `sam_vit_h_4b8939.pth` - Highest quality, slowest

Download your preferred model from [Meta's SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).

### Docker Configuration

The pipeline includes Docker settings for both ARM (M1/M2 Mac) and CUDA (NVIDIA GPU) environments. Modify the `docker_settings` in the code to switch between these configurations.

## Troubleshooting

### CUDA Issues

If you're using an NVIDIA GPU and encounter CUDA errors:
- Ensure your CUDA drivers are properly installed
- Check CUDA/PyTorch compatibility
- Switch to the CUDA Docker configuration by modifying the pipeline Docker settings

### M1/M2 Mac Issues

For Apple Silicon Macs:
- Ensure you're using the ARM-compatible Docker image
- PyTorch's MPS acceleration is used automatically when available

### Memory Issues

If you encounter memory errors:
- Reduce the batch size (process fewer images)
- Use a smaller SAM model variant
- Reduce the image resolution in the `load_and_prepare_data` step

## Extending the Pipeline

This pipeline can be extended in several ways:
- Add real ground truth masks instead of synthetic ones
- Implement more advanced fine-tuning techniques
- Add data augmentation for better generalization
- Enhance the HTML visualization with additional features

## License

Apache 2.0

## Acknowledgments

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta Research
- [ZenML](https://github.com/zenml-io/zenml) for pipeline management