from marshmallow import Schema, fields, validate

class UserRequestDTO(Schema):
    name = fields.Str(required=True, validate=validate.Length(min=2, max=50))
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=validate.Length(min=6))

class ChatRequestDTO(Schema):
    message = fields.Str(required=True) 

class SeedDataRequestDTO(Schema):
    url = fields.Str(required=True) 