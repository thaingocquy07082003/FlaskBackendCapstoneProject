from flask import Flask, request, jsonify , send_file
from flask_cors import CORS
from dtos.request_dto import UserRequestDTO, ChatRequestDTO , SeedDataRequestDTO
from dtos.response_dto import UserResponseDTO, ChatResponseDTO
from marshmallow import ValidationError
import openai
import os
from dotenv import load_dotenv
from LLM_Model.seed_data import seed_milvus, seed_milvus_live
from LLM_Model.agent import get_retriever as get_openai_retriever, get_llm_and_agent as get_openai_agent
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image ,ExifTags
import json
import tempfile
from datetime import datetime
from typing import List, Dict
import tiktoken
from marshmallow import Schema, fields, ValidationError
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
model = None
CORS(app, resources={r"/api/*": {"origins": "*"}})

def load_model():
    """Load the trained YOLO model"""
    global model
    MODEL_PATH = "best.pt"
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            import glob
            models = glob.glob("*/best.pt")
            if models:
                MODEL_PATH = models[-1]
                model = YOLO(MODEL_PATH)
                print(f"Model loaded from {MODEL_PATH}")
            else:
                raise Exception("No trained model found!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def draw_palm_keypoints(img, keypoints, boxes):
    """Draw palm line keypoints and bounding boxes"""
    img_draw = img.copy()
    
    colors = {
        0: (0, 0, 255),    # fate - red
        1: (0, 255, 0),    # head - green  
        2: (255, 0, 0),    # heart - blue
        3: (0, 255, 255)   # life - yellow (cyan in BGR)
    }
    
    class_names = ['fate', 'head', 'heart', 'life']
    
    for obj_idx in range(len(keypoints)):
        if boxes is not None and len(boxes) > obj_idx:
            box = boxes[obj_idx]
            class_id = int(box[5]) if len(box) > 5 else obj_idx % 4
            conf = box[4] if len(box) > 4 else 0.0
            
            x1, y1, x2, y2 = map(int, box[:4])
            color = colors.get(class_id, (255, 255, 255))
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_names[class_id]}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_draw, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img_draw, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            class_id = obj_idx % 4
            color = colors.get(class_id, (255, 255, 255))
        
        kpts = keypoints[obj_idx]
        for i, (x, y, conf) in enumerate(kpts):
            if conf > 0.5:
                x, y = int(x), int(y)
                cv2.circle(img_draw, (x, y), 4, color, -1)
                cv2.circle(img_draw, (x, y), 6, (255, 255, 255), 1)
                cv2.putText(img_draw, str(i), (x+8, y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        visible_points = [(int(x), int(y)) for x, y, conf in kpts if conf > 0.5]
        for i in range(len(visible_points) - 1):
            cv2.line(img_draw, visible_points[i], visible_points[i+1], color, 2)
    
    return img_draw

def get_palm_predictions(keypoints, boxes):
    """Get palmistry predictions based on line features"""
    predictions = []
    class_names = ['fate', 'head', 'heart', 'life']
    
    for obj_idx in range(len(keypoints)):
        if boxes is not None and len(boxes) > obj_idx:
            box = boxes[obj_idx]
            class_id = int(box[5]) if len(box) > 5 else obj_idx % 4
            conf = box[4] if len(box) > 4 else 0.0
            
            kpts = keypoints[obj_idx]
            visible_points = [(x, y) for x, y, conf in kpts if conf > 0.5]
            
            if len(visible_points) >= 2:
                # Calculate line features
                start_point = visible_points[0]
                end_point = visible_points[-1]
                length = np.sqrt((end_point[0]-start_point[0])**2 + (end_point[1]-start_point[1])**2)
                
                total_angle_change = 0
                for i in range(1, len(visible_points)-1):
                    v1 = np.array(visible_points[i]) - np.array(visible_points[i-1])
                    v2 = np.array(visible_points[i+1]) - np.array(visible_points[i])
                    angle = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
                    total_angle_change += angle
                
                line_type = class_names[class_id]
                shape_desc = "straight" if total_angle_change < 0.5 else "curved"
                length_desc = "long" if length > 300 else "medium" if length > 150 else "short"
                
                # Palmistry interpretations
                interpretation = ""
                if line_type == "life":
                    if length > 300:
                        interpretation = "Bạn có sức khỏe tốt và năng lượng dồi dào. Đường sinh mệnh dài cho thấy khả năng phục hồi tốt."
                    else:
                        interpretation = "Bạn cần chú ý hơn đến sức khỏe. Đường sinh mệnh ngắn không có nghĩa là yếu, mà chỉ cần quan tâm nhiều hơn."
                
                elif line_type == "head":
                    if length > 280 and shape_desc == "straight":
                        interpretation = "Bạn có tư duy logic mạnh mẽ và khả năng tập trung cao."
                    elif shape_desc == "curved":
                        interpretation = "Bạn có trí tưởng tượng phong phú và tư duy sáng tạo."
                    else:
                        interpretation = "Bạn có cách suy nghĩ thực tế và cân bằng."
                
                elif line_type == "heart":
                    if length > 250:
                        interpretation = "Bạn là người sống tình cảm, dễ đồng cảm với người khác."
                    else:
                        interpretation = "Bạn có cách tiếp cận cân bằng giữa lý trí và tình cảm."
                
                elif line_type == "fate":
                    if length > 200 and shape_desc == "straight":
                        interpretation = "Sự nghiệp của bạn ổn định và có định hướng rõ ràng."
                    elif shape_desc == "curved":
                        interpretation = "Bạn có nhiều thay đổi trong sự nghiệp nhưng sẽ đạt được thành công."
                    else:
                        interpretation = "Bạn sẽ tự tạo ra cơ hội cho bản thân thay vì phụ thuộc vào số phận."
                
                predictions.append({
                    "line_type": line_type,
                    "interpretation": interpretation,
                    "confidence": float(conf),
                    "length": float(length),
                    "shape": shape_desc
                })
    
    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)
                if orientation_value == 3:
                    image = image.rotate(180, expand=True)
                elif orientation_value == 6:
                    image = image.rotate(270, expand=True)
                elif orientation_value == 8:
                    image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (640, 640), interpolation=cv2.INTER_AREA)
        
        # Run inference
        results = model(img_cv)
        
        # Process results
        response_data = {
            "predictions": [],
            "image_info": {
                "width": img_cv.shape[1],
                "height": img_cv.shape[0],
                "channels": img_cv.shape[2]
            },
            "palm_predictions": []  # Thêm trường mới cho dự đoán bói toán
        }
        
        annotated_img = img_cv.copy()
        
        for r in results:
            if r.keypoints is not None and r.boxes is not None:
                keypoints = r.keypoints.data.cpu().numpy()
                boxes = r.boxes.data.cpu().numpy()
                
                # Draw annotations
                annotated_img = draw_palm_keypoints(annotated_img, keypoints, boxes)
                
                # Prepare response data
                for i in range(len(keypoints)):
                    for j in range(len(keypoints[i])):
                        response_data["predictions"].extend([
                            float(keypoints[i][j][0]),
                            float(keypoints[i][j][1])
                        ])
                
                # Get palmistry predictions
                response_data["palm_predictions"] = get_palm_predictions(keypoints, boxes)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response_data["annotated_image"] = img_base64
        response_data["total_xy"] = len(response_data["predictions"])
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ... (giữ nguyên các phần khác của code)

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    
    if model is None:
        print("Warning: Model could not be loaded. API will return errors.")
    app.run(host='0.0.0.0', port=5000, debug=True)