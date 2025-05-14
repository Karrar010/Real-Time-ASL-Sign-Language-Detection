import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet50

# Set page config
st.set_page_config(
    page_title="ASL Sign Language Detection",
    page_icon="👋",
    layout="wide"
)

# Define the class labels
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

def load_model():
    # Initialize model architecture
    model = resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_LABELS))
    
    # Load trained weights
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to PIL Image and apply transforms
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image)
    return image.unsqueeze(0)

def main():
    # Page title and description
    st.title("Real-Time ASL Sign Language Detection")
    st.write("Show hand signs to the camera to see real-time ASL detection")
    
    # Load model
    @st.cache_resource
    def get_model():
        return load_model()
    
    model = get_model()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    with col1:
        # Create a placeholder for the video feed
        video_placeholder = st.empty()
        
    with col2:
        # Create a placeholder for the prediction
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        # Add a confidence threshold slider
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
    
    # Add stop button
    stop_button = st.button("Stop")
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Create a copy for detection
        detection_frame = frame.copy()
        
        # Preprocess the frame
        input_tensor = preprocess_image(detection_frame)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            # Convert to Python scalars
            confidence = confidence.item()
            predicted_class = CLASS_LABELS[prediction.item()]
        
        # Display frame with prediction if confidence is above threshold
        if confidence > confidence_threshold:
            # Draw prediction on frame
            cv2.putText(
                frame,
                f"Sign: {predicted_class}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update video feed
        video_placeholder.image(frame_rgb, channels="RGB")
        
        # Update prediction and confidence
        prediction_placeholder.markdown(f"### Detected Sign: {predicted_class}")
        confidence_placeholder.markdown(f"### Confidence: {confidence:.2%}")
    
    # Release resources when stopped
    cap.release()
    st.write("Stopped")

if __name__ == "__main__":
    main()
