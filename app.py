import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import segmentation_models_pytorch as smp
import cv2
import os

# --- 1. SETTINGS & CONFIG ---
st.set_page_config(page_title="Retina AI", layout="wide", initial_sidebar_state="expanded")

# Clinical Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { 
        background-color: #161b22; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #30363d;
    }
    div[data-testid="stStatus"] { background-color: #161b22; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VISION_MODEL_PATH = "models/vision_model_unet.pth"
GRADING_MODEL_PATH = "models/grading_model_effnet.pth"

SEVERITY_NAMES = {
    0: "Normal", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "PDR (Proliferative)"
}

# --- 2. GRAD-CAM CLASS ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        def save_activation(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        one_hot = torch.zeros((1, output.size()[-1]), device=DEVICE)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (512, 512))
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_ai_agents():
    v_model = smp.Unet(encoder_name="resnet34", classes=1).to(DEVICE)
    if os.path.exists(VISION_MODEL_PATH):
        v_model.load_state_dict(torch.load(VISION_MODEL_PATH, map_location=DEVICE))
    v_model.eval()
    
    g_model = models.efficientnet_b0()
    g_model.classifier[1] = nn.Linear(g_model.classifier[1].in_features, 5)
    if os.path.exists(GRADING_MODEL_PATH):
        g_model.load_state_dict(torch.load(GRADING_MODEL_PATH, map_location=DEVICE))
    g_model.eval()
    return v_model, g_model

# --- 4. MAIN INTERFACE ---
st.title("👁️ Retina AI: Professional Diagnostic Suite")
st.caption("AI-Powered Retinal Pathology Analysis v2.0")

st.sidebar.title("🩺 Clinical Control")
uploaded_file = st.sidebar.file_uploader("Upload Retinal Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    v_agent, g_agent = load_ai_agents()
    raw_img = Image.open(uploaded_file).convert("RGB")
    prep = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    input_tensor = prep(raw_img).unsqueeze(0).to(DEVICE)

    with st.status("🩺 Initializing Neural Diagnostic Agents...", expanded=True) as status:
        st.write("Extracting retinal features...")
        
        # 1. Vision Inference
        with torch.no_grad():
            mask = torch.sigmoid(v_agent(input_tensor)).cpu().squeeze().numpy() > 0.5
        
        # 2. Grading Inference
        cam_engine = GradCAM(g_agent, g_agent.features[-1])
        logits = g_agent(input_tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        
        # 3. XAI Generation
        heatmap = cam_engine.generate(input_tensor, pred_idx.item())
        
        status.update(label="✅ Diagnosis Complete!", state="complete", expanded=False)

    # --- THE "PRO" DASHBOARD LAYOUT ---
    st.divider()
    res_col, met_col = st.columns([2, 1])
    
    with res_col:
        severity = SEVERITY_NAMES[pred_idx.item()]
        if pred_idx.item() >= 2:
            st.error(f"## **Diagnosis: {severity}**")
        else:
            st.success(f"## **Diagnosis: {severity}**")
            
    with met_col:
        st.metric("AI Confidence Score", f"{conf.item()*100:.1f}%")

    st.subheader("🔍 Integrated Evidence Reports")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.info("**Raw Input**")
        st.image(raw_img, use_container_width=True)
    with c2:
        st.info("**Lesion Map**")
        fig1, ax1 = plt.subplots(facecolor='black')
        ax1.imshow(mask, cmap='hot')
        ax1.axis('off')
        st.pyplot(fig1)
    with c3:
        st.info("**XAI Validation**")
        img_np = np.array(raw_img.resize((512, 512)))
        fig2, ax2 = plt.subplots(facecolor='black')
        ax2.imshow(img_np)
        ax2.imshow(heatmap, cmap='jet', alpha=0.5) 
        ax2.axis('off')
        st.pyplot(fig2)

    with st.expander("📄 Clinical Recommendation & Metadata"):
        if pred_idx.item() >= 2:
            st.write("**Recommended Action:** Immediate referral to a specialist for further imaging.")
        else:
            st.write("**Recommended Action:** Routine annual screening.")
        st.write(f"**Model Specs:** EfficientNet-B0 + U-Net ResNet34 | **Device:** {DEVICE}")

else:
    st.info("👈 Please upload a retinal image in the sidebar to begin.")