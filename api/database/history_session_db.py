from mongoengine import EmbeddedDocument, StringField, DateTimeField
import datetime

class HistorySession(EmbeddedDocument):
    date = DateTimeField(default=datetime.datetime.utcnow)
    user_input = StringField()
    output = StringField()
    prev_conv = StringField()
    relevant_conv = StringField()
    all_prompt = StringField()