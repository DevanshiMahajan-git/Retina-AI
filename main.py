import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
import segmentation_models_pytorch as smp

# ==========================================
# 1. SETTINGS & PATHS (Update these!)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_FOLDER = "data/Grading_Images"  # Path to your processed 512x512 images
CSV_PATH = "data/labels.csv"          # Path to your labels CSV
VISION_MODEL_PATH = "models/vision_model_unet.pth"
GRADING_MODEL_PATH = "models/grading_model_effnet.pth"

SEVERITY_NAMES = {
    0: "Normal",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR (Proliferative)"
}

# The same transformation used during Colab training
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# ==========================================
# 2. LOAD MODELS
# ==========================================
def load_agents():
    print(f"--- Loading Agents on {DEVICE} ---")
    
    # Vision Agent (U-Net)
    v_model = smp.Unet(encoder_name="resnet34", classes=1).to(DEVICE)
    v_model.load_state_dict(torch.load(VISION_MODEL_PATH, map_location=DEVICE))
    v_model.eval()
    
    # Senior Doctor Agent (EfficientNet)
    g_model = models.efficientnet_b0()
    num_ftrs = g_model.classifier[1].in_features
    g_model.classifier[1] = nn.Linear(num_ftrs, 5)
    g_model.load_state_dict(torch.load(GRADING_MODEL_PATH, map_location=DEVICE))
    g_model.eval()
    
    return v_model, g_model

# ==========================================
# 3. COMPOUND DIAGNOSTIC FUNCTION
# ==========================================
def run_diagnostic(image_id, true_grade, v_model, g_model):
    # Prepare Image
    img_path = os.path.join(IMAGE_FOLDER, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return

    raw_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True # Required for XAI heatmap

    # Agent 1: Vision (Lesion Map)
    with torch.no_grad():
        seg_output = v_model(input_tensor)
        seg_mask = torch.sigmoid(seg_output).cpu().squeeze().numpy() > 0.5

    # Agent 2: ML (Severity Grading)
    grading_output = g_model(input_tensor)
    prob = F.softmax(grading_output, dim=1)
    conf, pred_idx = torch.max(prob, 1)

    # Agent 3: XAI (Decision Heatmap)
    # Using GradCAM: finding which pixels the model cares about for the predicted class
    grading_output[0, pred_idx].backward()
    saliency, _ = torch.max(torch.abs(input_tensor.grad.data.cpu().squeeze()), dim=0)

    # --- GENERATE DASHBOARD ---
    plt.figure(figsize=(20, 6))
    
    # Plot 1: Original
    plt.subplot(1, 4, 1)
    plt.title(f"Original Eye\n({image_id})")
    plt.imshow(raw_img)
    plt.axis('off')

    # Plot 2: Vision Agent
    plt.subplot(1, 4, 2)
    plt.title("Vision Agent:\nLesion Map")
    plt.imshow(seg_mask, cmap='hot')
    plt.axis('off')

    # Plot 3: XAI Heatmap
    plt.subplot(1, 4, 3)
    plt.title("XAI Agent:\nDecision Evidence")
    plt.imshow(saliency, cmap='jet')
    plt.axis('off')

    # Plot 4: Final Report
    plt.subplot(1, 4, 4)
    plt.axis('off')
    color = 'green' if pred_idx.item() == true_grade else 'red'
    
    plt.text(0.5, 0.7, "FINAL DIAGNOSIS:", fontsize=12, ha='center', weight='bold')
    plt.text(0.5, 0.5, SEVERITY_NAMES[pred_idx.item()], fontsize=18, ha='center', color=color, weight='bold')
    plt.text(0.5, 0.35, f"Confidence: {conf.item()*100:.1f}%", fontsize=12, ha='center')
    plt.text(0.5, 0.15, f"Reference Truth: {SEVERITY_NAMES[true_grade]}", fontsize=10, ha='center', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Load the Database
    df = pd.read_csv(CSV_PATH).iloc[:, [0, 1]]
    df.columns = ['Image_ID', 'Grade']
    
    # Load the Brains
    vision_agent, doctor_agent = load_agents()
    
    # Run a sample (Change index to test different images)
    sample_index = 45 
    img_id = df.iloc[sample_index, 0]
    actual_grade = df.iloc[sample_index, 1]
    
    run_diagnostic(img_id, actual_grade, vision_agent, doctor_agent)