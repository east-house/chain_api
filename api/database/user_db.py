from mongoengine import Document, StringField, DictField, DateTimeField, ListField, EmbeddedDocumentField
import datetime

from api.m2m_llm.database import HistorySession

class User(Document):
    user_id = StringField(required=True, unique=True)
    history_sessions = DictField(ListField(EmbeddedDocumentField(HistorySession)))
    date_modified = DateTimeField(default=datetime.datetime.utcnow)