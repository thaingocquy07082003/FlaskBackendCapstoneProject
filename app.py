from flask import Flask, request, jsonify , send_file
from flask_cors import CORS
from dtos.request_dto import UserRequestDTO, ChatRequestDTO , SeedDataRequestDTO
from dtos.response_dto import UserResponseDTO, ChatResponseDTO
from marshmallow import ValidationError
import openai
import os
from dotenv import load_dotenv
from LLM_Model.seed_data import seed_milvus, seed_milvus_live  # Hàm xử lý dữ liệu
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
            # Try to find any available model
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
    
    # Define colors for different palm lines
    colors = {
        0: (0, 0, 255),    # fate - red
        1: (0, 255, 0),    # head - green  
        2: (255, 0, 0),    # heart - blue
        3: (0, 255, 255)   # life - yellow (cyan in BGR)
    }
    
    class_names = ['fate', 'head', 'heart', 'life']
    
    for obj_idx in range(len(keypoints)):
        # Get class from boxes
        if boxes is not None and len(boxes) > obj_idx:
            box = boxes[obj_idx]
            class_id = int(box[5]) if len(box) > 5 else obj_idx % 4
            conf = box[4] if len(box) > 4 else 0.0
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            color = colors.get(class_id, (255, 255, 255))
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # Draw class label
            label = f"{class_names[class_id]}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img_draw, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img_draw, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            class_id = obj_idx % 4
            color = colors.get(class_id, (255, 255, 255))
        
        # Draw keypoints
        kpts = keypoints[obj_idx]
        for i, (x, y, conf) in enumerate(kpts):
            if conf > 0.5:  # Only draw visible keypoints
                x, y = int(x), int(y)
                cv2.circle(img_draw, (x, y), 4, color, -1)
                cv2.circle(img_draw, (x, y), 6, (255, 255, 255), 1)
                cv2.putText(img_draw, str(i), (x+8, y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw connections between consecutive keypoints
        visible_points = [(int(x), int(y)) for x, y, conf in kpts if conf > 0.5]
        for i in range(len(visible_points) - 1):
            cv2.line(img_draw, visible_points[i], visible_points[i+1], color, 2)
    
    return img_draw

@app.route('/api/users', methods=['GET'])
def get_users():
    # Example GET endpoint
    try:
        # In a real application, you would fetch data from a database
        response_data = UserResponseDTO.create(
            id=1,
            name="Example User",
            email="user@example.com"
        )
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/users', methods=['POST'])
def create_user():
    # Example POST endpoint
    try:
        data = request.get_json()
        # Validate request data using RequestDTO
        user_request = UserRequestDTO()
        validated_data = user_request.load(data)
        
        # In a real application, you would save the data to a database
        response_data = UserResponseDTO.create(
            id=1,
            name=validated_data['name'],
            email=validated_data['email']
        )
        return jsonify(response_data), 201
    except ValidationError as e:
        return jsonify({"error": e.messages}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/api/chat', methods=['POST'])
# def chat():
#     try:
#         data = request.get_json()
#         # Validate request data using ChatRequestDTO
#         chat_request = ChatRequestDTO()
#         validated_data = chat_request.load(data)
#         retriever = get_openai_retriever()
#         agent_executor = get_openai_agent(retriever, "gpt4")
#         response = agent_executor.invoke(
#             {
#                 "input": validated_data['message'],
#                 # "chat_history": chat_history
#             },
#         )
       
#         # Create response using ChatResponseDTO
#         response_data = ChatResponseDTO.create(
#             response=response["output"]
#         )
#         return jsonify(response_data), 200
#     except ValidationError as e:
#         return jsonify({"error": e.messages}), 400
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# Hàm đếm token
def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text, allowed_special="all"))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

# Hàm tinh giảm văn bản
def truncate_text(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    if not text:
        return ""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text, allowed_special="all")
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

# Hàm tinh giảm tài liệu từ retriever
def truncate_retriever_documents(retriever, query: str, max_tokens: int, model: str = "gpt-4") -> List[Document]:
    if not query:
        return []
    try:
        documents = retriever.get_relevant_documents(query)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []
    
    total_tokens = sum(count_tokens(doc.page_content) for doc in documents)
    
    if total_tokens <= max_tokens:
        return documents
    
    truncated_docs = []
    current_tokens = 0
    
    for doc in sorted(documents, key=lambda x: count_tokens(x.page_content)):  # Sắp xếp từ ngắn đến dài
        doc_tokens = count_tokens(doc.page_content)
        if current_tokens + doc_tokens <= max_tokens:
            truncated_docs.append(doc)
            current_tokens += doc_tokens
        else:
            max_doc_tokens = max_tokens - current_tokens
            if max_doc_tokens > 50:  # Đảm bảo tài liệu có ít nhất 50 token
                encoding = tiktoken.encoding_for_model(model)
                truncated_text = encoding.decode(encoding.encode(doc.page_content, allowed_special="all")[:max_doc_tokens])
                truncated_docs.append(Document(page_content=truncated_text, metadata=doc.metadata))
                current_tokens += max_doc_tokens
            break
    
    return truncated_docs

# Hàm chuẩn bị đầu vào với tinh giảm token
def prepare_input(validated_data: Dict, retriever, max_tokens: int = 8000, model: str = "gpt-4") -> Dict:
    # Kiểm tra validated_data và message
    if not isinstance(validated_data, dict) or 'message' not in validated_data:
        return {"input": ""}  # Trả về input rỗng nếu dữ liệu không hợp lệ
    
    message = validated_data.get('message', '')
    if not message:
        return {"input": ""}  # Trả về input rỗng nếu message rỗng
    
    total_tokens = count_tokens(message)
    
    # Nếu có chat_history (khi bạn bỏ comment)
    chat_history = []  # Giả sử rỗng nếu chưa sử dụng
    # total_tokens += sum(count_tokens(msg.content) for msg in chat_history)
    
    # Lấy và tinh giảm tài liệu từ retriever
    retrieved_content = ""
    if retriever and total_tokens < max_tokens:
        max_retriever_tokens = max_tokens - total_tokens - 100  # Dự phòng 100 token
        retrieved_docs = truncate_retriever_documents(retriever, message, max_retriever_tokens, model)
        retrieved_content = "\n".join([doc.page_content for doc in retrieved_docs if doc.page_content])
        total_tokens += count_tokens(retrieved_content)
        
        # Thêm nội dung từ retriever vào message
        if retrieved_content:
            message = f"{message}\n\nRetrieved context:\n{retrieved_content}"
    
    # Tinh giảm message nếu vượt quá giới hạn
    if total_tokens > max_tokens:
        message = truncate_text(message, max_tokens - 100, model)  # Dự phòng 100 token
    
    return {"input": message}

# Route API
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Lấy và validate dữ liệu
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        chat_request = ChatRequestDTO()
        validated_data = chat_request.load(data)
        
        # Lấy retriever và agent
        retriever = get_openai_retriever()  # Giả sử hàm này trả về EnsembleRetriever
        if retriever is None:
            return jsonify({"error": "Failed to initialize retriever"}), 500
        
        agent_executor = get_openai_agent(retriever, "gpt4")
        if agent_executor is None:
            return jsonify({"error": "Failed to initialize agent"}), 500
        
        # Chuẩn bị đầu vào với tinh giảm token
        input_data = prepare_input(validated_data, retriever, max_tokens=8000)
        if not input_data["input"]:
            return jsonify({"error": "Input is empty after processing"}), 400
        
        # Gọi agent_executor
        response = agent_executor.invoke(input_data)
        if response is None or "output" not in response:
            return jsonify({"error": "Invalid response from agent"}), 500
        
        # Tạo phản hồi
        response_data = ChatResponseDTO.create(response=response["output"])
        return jsonify(response_data), 200
    
    except ValidationError as e:
        return jsonify({"error": e.messages}), 400
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

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
    
@app.route('/seeddata', methods=['POST'])
def SeedData():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        # Validate request data using ChatRequestDTO
        seed_data_request = SeedDataRequestDTO()
        validated_data = seed_data_request.load(data)
        seed_milvus_live(validated_data['url'], 'http://54.253.52.57:19530', 'data_test', 'data from web', use_ollama=False)
        return jsonify('successful seed data'), 200
    except ValidationError as e:
        return jsonify({"error": e.messages}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    
    if model is None:
        print("Warning: Model could not be loaded. API will return errors.")
    app.run(host='0.0.0.0', port=5000, debug=True) 
