"""
ZenML pipeline for SAM fine-tuning with interactive HTML visualization
"""

import os
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.types import HTMLString
from segment_anything import SamPredictor, sam_model_registry
from typing import List, Dict

@step
def load_and_prepare_data(image_dir: str, max_images: int = 5):
    """Load images and create synthetic masks for fine-tuning."""
    images = []
    masks = []
    filenames = []
    
    # SAM expected image size
    image_size = (1024, 1024)
    # SAM mask decoder output size
    mask_size = (256, 256)
    
    # Load images
    for f in os.listdir(image_dir)[:max_images]:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_dir, f)
            # Resize image to fit SAM's expected dimensions
            img_pil = Image.open(path).convert("RGB").resize(image_size)
            img = np.array(img_pil)
            images.append(img)
            filenames.append(f)
            
            # Create a simple synthetic circular mask for training (at 256x256 resolution)
            h, w = mask_size
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h//2, w//2
            radius = min(h, w) // 4
            mask = ((y - center_y)**2 + (x - center_x)**2 <= radius**2).astype(np.float32)
            masks.append(mask)
    
    return {"images": images, "masks": masks, "filenames": filenames}

@step
def finetune_sam(data, checkpoint_path: str, model_type: str = "vit_b", epochs: int = 3):
    """Load SAM model and fine-tune it on the provided data."""
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    
    # Setup optimizer (only fine-tune mask decoder for simplicity)
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-5)
    
    print("Starting fine-tuning...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for idx, (image, mask) in enumerate(zip(data["images"], data["masks"])):
            # Convert to tensors
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # Get image embedding (no gradients needed here)
            with torch.no_grad():
                image_embedding = sam.image_encoder(image_tensor)
            
            # Create a center point prompt
            h, w = mask.shape
            input_point = torch.tensor([[[w//2, h//2]]], device=device)
            input_label = torch.tensor([[1]], device=device)
            
            # Get prompt embeddings
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=(input_point, input_label),
                boxes=None,
                masks=None,
            )
            
            # Predict masks
            mask_predictions, _ = sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # Calculate loss and update weights
            loss = F.binary_cross_entropy_with_logits(mask_predictions, mask_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss/len(data['images']):.4f}")
    
    print("Fine-tuning completed")
    return sam

@step
def generate_masks(model, data, output_dir: str) -> Dict[str, List[str]]:
    """Generate and save segmentation masks with overlay visualization."""
    os.makedirs(output_dir, exist_ok=True)
    predictor = SamPredictor(model)
    
    # Track all output paths
    result_data = {
        "original": [],
        "mask": [],
        "overlay": [],
        "filenames": data["filenames"]
    }
    
    for img, filename in zip(data["images"], data["filenames"]):
        # Save original image
        orig_path = os.path.join(output_dir, f"original_{filename}")
        Image.fromarray(img).save(orig_path)
        result_data["original"].append(orig_path)
        
        # Set image in predictor
        predictor.set_image(img)
        
        # Generate mask from center point prompt
        h, w = img.shape[:2]
        masks, _, _ = predictor.predict(
            point_coords=np.array([[w//2, h//2]]),
            point_labels=np.array([1]),
            multimask_output=False
        )
        
        # Get the binary mask (upscaled from model output)
        mask = masks[0]
        
        # Save the binary mask
        mask_path = os.path.join(output_dir, f"mask_{filename}")
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        result_data["mask"].append(mask_path)
        
        # Create overlay image with green highlight
        overlay_img = img.copy()
        overlay_img = np.where(
            np.expand_dims(mask, axis=2),
            np.clip(overlay_img * 0.7 + np.array([0, 200, 0]) * 0.3, 0, 255),
            overlay_img
        ).astype(np.uint8)
        
        # Save the overlay image
        overlay_path = os.path.join(output_dir, f"overlay_{filename}")
        Image.fromarray(overlay_img).save(overlay_path)
        result_data["overlay"].append(overlay_path)
    
    return result_data

@step
def create_interactive_html_artifact(result_data: Dict[str, List[str]]) -> HTMLString:
    """Create an interactive HTML visualization for comparing segmentation results."""
    # Convert images to base64 for embedding in HTML
    def img_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    # Create image data for the HTML
    images_data = []
    for i, filename in enumerate(result_data["filenames"]):
        img_data = {
            "name": filename,
            "original": img_to_base64(result_data["original"][i]),
            "mask": img_to_base64(result_data["mask"][i]),
            "overlay": img_to_base64(result_data["overlay"][i])
        }
        images_data.append(img_data)
    
    # HTML template for visualization
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SAM Segmentation Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                text-align: center;
                color: #333;
            }
            .controls {
                display: flex;
                justify-content: center;
                margin: 20px 0;
                gap: 15px;
            }
            select {
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #ddd;
            }
            .comparison {
                display: flex;
                gap: 15px;
                justify-content: center;
                flex-wrap: wrap;
            }
            .image-card {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 15px;
                width: 45%;
                min-width: 300px;
            }
            .card-title {
                font-weight: bold;
                margin-bottom: 10px;
            }
            img {
                width: 100%;
                border-radius: 4px;
                max-height: 350px;
                object-fit: contain;
            }
        </style>
    </head>
    <body>
        <h1>SAM Fine-tuning Results</h1>
        
        <div class="controls">
            <label for="imageSelector">Select Image:</label>
            <select id="imageSelector" onchange="updateView()">
                <!-- Will be populated by JavaScript -->
            </select>
            
            <label for="rightViewSelector">Compare with:</label>
            <select id="rightViewSelector" onchange="updateView()">
                <option value="original">Original Image</option>
                <option value="mask" selected>Binary Mask</option>
                <option value="overlay">Overlay</option>
            </select>
        </div>
        
        <div class="comparison">
            <div class="image-card">
                <div class="card-title">Original Image</div>
                <img id="leftImage" src="" alt="Original image">
            </div>
            
            <div class="image-card">
                <div class="card-title" id="rightTitle">Binary Mask</div>
                <img id="rightImage" src="" alt="Comparison image">
            </div>
        </div>
        
        <script>
            // Store image data
            const imagesData = IMAGES_DATA_PLACEHOLDER;
            
            function initializeUI() {
                const imageSelector = document.getElementById('imageSelector');
                
                // Add options for each image
                imagesData.forEach((imgData, index) => {
                    const option = document.createElement('option');
                    option.value = index;
                    option.textContent = imgData.name;
                    imageSelector.appendChild(option);
                });
                
                // Initial update
                updateView();
            }
            
            function updateView() {
                const imageIndex = document.getElementById('imageSelector').value;
                const rightView = document.getElementById('rightViewSelector').value;
                
                const imgData = imagesData[imageIndex];
                
                // Update images
                document.getElementById('leftImage').src = `data:image/jpeg;base64,${imgData.original}`;
                document.getElementById('rightImage').src = `data:image/jpeg;base64,${imgData[rightView]}`;
                
                // Update title
                document.getElementById('rightTitle').textContent = 
                    rightView === 'original' ? 'Original Image' : 
                    rightView === 'mask' ? 'Binary Mask' : 'Overlay';
            }
            
            // Initialize when page loads
            window.onload = initializeUI;
        </script>
    </body>
    </html>
    """
    
    # Replace placeholder with actual image data
    import json
    html_content = html_content.replace('IMAGES_DATA_PLACEHOLDER', json.dumps(images_data))
    
    # Return HTML as a special HTMLString type for ZenML to visualize
    return HTMLString(html_content)

# Simplified Docker settings
docker_settings = DockerSettings(
    parent_image="pytorch/pytorch:1.13.1-runtime",
    apt_packages=["libgl1-mesa-glx", "libglib2.0-0"],
    requirements=[
        "zenml",
        "segment-anything", 
        "pillow",
        "numpy"
    ]
)

@pipeline(settings={"docker": docker_settings})
def sam_pipeline(image_dir, checkpoint_path, output_dir, epochs=2):
    """SAM fine-tuning pipeline with interactive HTML visualization."""
    data = load_and_prepare_data(image_dir)
    model = finetune_sam(data, checkpoint_path, epochs=epochs)
    result_data = generate_masks(model, data, output_dir)
    create_interactive_html_artifact(result_data)

# Example usage
if __name__ == "__main__":
    sam_pipeline(
        image_dir="./images",
        checkpoint_path="./sam_vit_b_01ec64.pth",
        output_dir="./output",
        epochs=2
    )
