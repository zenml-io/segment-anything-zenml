#!/bin/bash
set -e  # Exit on error

echo "Setting up SAM ZenML Pipeline environment..."

# Create project directory structure
PROJECT_DIR="$HOME/sam_zenml_project"
echo "Creating project directory at $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
mkdir -p images
mkdir -p output

# Download sample images for testing
echo "Downloading sample images..."
wget -P images/ https://images.unsplash.com/photo-1543466835-00a7907e9de1?auto=format&fit=crop&w=500 -O images/dog.jpg
wget -P images/ https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?auto=format&fit=crop&w=500 -O images/cat.jpg
wget -P images/ https://images.unsplash.com/photo-1557800636-894a64c1696f?auto=format&fit=crop&w=500 -O images/person.jpg
wget -P images/ https://images.unsplash.com/photo-1541167760496-1628856ab772?auto=format&fit=crop&w=500 -O images/building.jpg
wget -P images/ https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?auto=format&fit=crop&w=500 -O images/flower.jpg

# Download SAM checkpoint (ViT-B variant)
echo "Downloading SAM model checkpoint (vit_b variant)..."
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Set up Python environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install zenml segment-anything torch pillow numpy

# Initialize ZenML
echo "Initializing ZenML..."
zenml init

# Create the Python script file
echo "Creating the SAM ZenML pipeline script..."
cat > sam_pipeline.py << 'EOF'
"""
ZenML pipeline for SAM model with a simplified but realistic fine-tuning process.
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from zenml import pipeline, step
from zenml.config import DockerSettings
from segment_anything import SamPredictor, sam_model_registry
import random

@step
def load_and_prepare_data(image_dir: str, max_images: int = 5):
    """Load images and create synthetic masks for fine-tuning."""
    images = []
    masks = []
    paths = []
    
    # Load images
    for f in os.listdir(image_dir)[:max_images]:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_dir, f)
            img = np.array(Image.open(path).convert("RGB"))
            images.append(img)
            paths.append(path)
            
            # Create a synthetic binary mask for training
            # In real scenarios, you would use ground truth masks
            h, w = img.shape[:2]
            # Create a circular mask in center of image (simplified)
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h//2, w//2
            radius = min(h, w) // 4
            mask = ((y - center_y)**2 + (x - center_x)**2 <= radius**2).astype(np.float32)
            masks.append(mask)
    
    return {"images": images, "masks": masks, "paths": paths}

@step
def finetune_sam(data, checkpoint_path: str, model_type: str = "vit_b", epochs: int = 3):
    """Load and realistically fine-tune SAM model."""
    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon (M1/M2) GPU
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    
    # Configure training (only fine-tune the mask decoder part)
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    
    # Setup optimizer
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-5)
    
    # Simplified training loop
    print("Starting fine-tuning...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Process mini-batches (single image for simplicity)
        for idx, (image, mask) in enumerate(zip(data["images"], data["masks"])):
            # Convert to tensor
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Get image embedding
            with torch.no_grad():
                image_embedding = sam.image_encoder(image_tensor)
            
            # Run decoder with a random prompt point
            h, w = mask.shape
            prompt_point = torch.tensor([[[w//2 + random.randint(-30, 30), 
                                           h//2 + random.randint(-30, 30)]]], 
                                       device=device)
            prompt_label = torch.tensor([[1]], device=device)
            
            # Forward pass
            mask_predictions, _ = sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sam.prompt_encoder(
                    points=(prompt_point, prompt_label),
                    boxes=None,
                    masks=None,
                )[0],
                dense_prompt_embeddings=None,
                multimask_output=False,
            )
            
            # Calculate loss (binary cross entropy)
            loss = F.binary_cross_entropy_with_logits(mask_predictions, mask_tensor)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(data['images']):.4f}")
    
    print("Fine-tuning completed")
    return sam

@step
def generate_masks(model, data, output_dir: str):
    """Generate and save segmentation masks."""
    os.makedirs(output_dir, exist_ok=True)
    predictor = SamPredictor(model)
    
    for img, path in zip(data["images"], data["paths"]):
        # Set image
        predictor.set_image(img)
        
        # Center point prompt
        h, w = img.shape[:2]
        masks, _, _ = predictor.predict(
            point_coords=np.array([[w//2, h//2]]),
            point_labels=np.array([1]),
            multimask_output=False
        )
        
        # Save mask
        mask = masks[0]
        output_path = os.path.join(output_dir, f"mask_{os.path.basename(path)}")
        Image.fromarray((mask * 255).astype(np.uint8)).save(output_path)
    
    return {"status": "completed"}

# Docker settings for different environments
mac_m1_docker_settings = DockerSettings(
    parent_image="arm64v8/python:3.9-slim",
    apt_packages=["libgl1-mesa-glx", "libglib2.0-0"],
    requirements=[
        "zenml",
        "segment-anything",
        "torch",
        "pillow",
        "numpy"
    ]
)

cuda_docker_settings = DockerSettings(
    parent_image="pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime",
    apt_packages=["libgl1-mesa-glx", "libglib2.0-0"],
    requirements=[
        "zenml",
        "segment-anything",
        "pillow",
        "numpy"
    ]
)

# Pipeline for Mac M1
@pipeline(settings={"docker": mac_m1_docker_settings})
def sam_pipeline(image_dir, checkpoint_path, output_dir, epochs=3):
    """SAM pipeline with realistic fine-tuning."""
    data = load_and_prepare_data(image_dir)
    model = finetune_sam(data, checkpoint_path, epochs=epochs)
    generate_masks(model, data, output_dir)

# Example usage
if __name__ == "__main__":
    # Run pipeline with minimal epochs for faster execution
    sam_pipeline(
        image_dir="./images",
        checkpoint_path="./sam_vit_b_01ec64.pth",
        output_dir="./output",
        epochs=2  # Minimal fine-tuning for demonstration
    )
EOF

echo "=================================================="
echo "Setup complete! Here's how to run the pipeline:"
echo "1. Make sure you're in the project directory: cd $PROJECT_DIR"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the pipeline: python sam_pipeline.py"
echo "4. Check the 'output' directory for generated masks"
echo "=================================================="