# Flask API with DTOs

This is a Flask-based API that demonstrates the use of Request and Response DTOs (Data Transfer Objects).

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## API Endpoints

### GET /api/users
- Returns a list of users
- Response: UserResponseDTO

### POST /api/users
- Creates a new user
- Request Body: UserRequestDTO
- Response: UserResponseDTO

## DTOs Structure

- `dtos/request_dto.py`: Contains RequestDTO classes for validating incoming data
- `dtos/response_dto.py`: Contains ResponseDTO classes for formatting outgoing data 
