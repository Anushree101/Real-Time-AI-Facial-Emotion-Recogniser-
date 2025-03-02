import gradio as gr
import cv2
from deepface import DeepFace
import numpy as np

# Function to predict emotion from an image
def predict_emotion_image(image):
    try:
        # Convert the Gradio image (PIL format) to an OpenCV image
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Analyze the emotion using DeepFace
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        
        # Get the dominant emotion
        dominant_emotion = result[0]['dominant_emotion']  # Access the first result if it's a list
        
        return f"Detected Emotion: {dominant_emotion}"
    except Exception as e:
        return f"Error in image emotion detection: {str(e)}"

# Function to process video, detect faces, and predict emotions
def predict_emotion_video(video):
    try:
        cap = cv2.VideoCapture(video)
        
        if not cap.isOpened():
            return "Error: Unable to open video."
        
        # Initialize variables to store detected emotions
        detected_emotions = []
        
        # Define codec and create VideoWriter object to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        output_filename = "output_video.mp4"
        out = cv2.VideoWriter(output_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        
        # Processing each frame of the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                
                try:
                    result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = result[0]['dominant_emotion']
                    
                    # Append detected emotion for the current frame
                    detected_emotions.append(dominant_emotion)
                    
                    # Draw rectangle and label around face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                
                except Exception as e:
                    print(f"Error analyzing face in video: {e}")
            
            # Write the frame with the detection to the output video
            out.write(frame)

        # Release video resources
        cap.release()
        out.release()
        
        # Combine detected emotions into a result string
        result_text = "Video Emotion Results: " + ", ".join(detected_emotions) if detected_emotions else "No emotion detected."
        
        return result_text
    
    except Exception as e:
        return f"Error in video emotion detection: {str(e)}"

# Function to handle choice (image or video)
def process_choice(choice, image=None, video=None):
    if choice == "Image Emotion Detection":
        return predict_emotion_image(image)
    elif choice == "Video Emotion Detection":
        return predict_emotion_video(video)
    else:
        return "Please select a valid option."

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='color: #e91e63;'>Image and Video Emotion Recognition</h1>")
    
    # Dropdown to select between image or video detection
    choice = gr.Dropdown(["Image Emotion Detection", "Video Emotion Detection"], label="Choose Mode")
    
    # Inputs for image or video
    image_input = gr.Image(type="pil", label="Upload Image", visible=False)
    video_input = gr.Video(label="Upload Video", visible=False)
    
    # Function to show/hide inputs based on selection
    def update_input(choice):
        if choice == "Image Emotion Detection":
            return gr.update(visible=True), gr.update(visible=False)
        elif choice == "Video Emotion Detection":
            return gr.update(visible=False), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=False)

    # Update visibility of inputs
    choice.change(fn=update_input, inputs=choice, outputs=[image_input, video_input])
    
    # Output
    result_output = gr.Textbox(label="Emotion Detection Result")
    
    # Button to process the image or video
    submit_btn = gr.Button("Analyze Emotion")
    
    # Connect button to function for processing choice
    submit_btn.click(fn=process_choice, inputs=[choice, image_input, video_input], outputs=result_output)

# Launch the Gradio app
demo.launch()
