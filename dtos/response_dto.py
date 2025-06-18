from marshmallow import Schema, fields

class UserResponseDTO(Schema):
    id = fields.Int()
    name = fields.Str()
    email = fields.Str()

    @staticmethod
    def create(id, name, email):
        return {
            "id": id,
            "name": name,
            "email": email
        }

class ChatResponseDTO(Schema):
    response = fields.Str()

    @staticmethod
    def create(response):
        return {
            "response": response
        }

    def to_dict(self):
        return {
            "response": self.response
        } 