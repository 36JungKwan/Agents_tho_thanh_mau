import uuid
import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON, Uuid
from sqlalchemy.orm import declarative_base

Base = declarative_base()

def get_vn_time():
    vn_timezone = datetime.timezone(datetime.timedelta(hours=7))
    return datetime.datetime.now(vn_timezone)

# ==========================================
# PHẦN 1: ÁNH XẠ CÁC BẢNG ĐÃ CÓ TRONG DB CHÍNH
# (Khai báo để làm Khóa ngoại, không ảnh hưởng data cũ)
# ==========================================

class User(Base):
    __tablename__ = "users" # Tên chuẩn trong DB chính
    
    # Cột String (VARCHAR) theo đúng thiết kế db chính thức
    id = Column(String, primary_key=True) 
    name = Column(String, nullable=True)
    email = Column(String, unique=True, nullable=True)
    # (Các cột khác của DB chính vẫn an toàn, không cần khai báo hết ở đây)

class Artisan(Base):
    __tablename__ = "artists" # Trỏ đúng vào bảng artists
    
    # DB chính dùng UUID native
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    userid = Column(String, ForeignKey("users.id"), nullable=True) 
    
    # 2 Cột AI RAG
    bio = Column(Text, nullable=True) 
    style_profile = Column(Text, nullable=True)

# ==========================================
# PHẦN 2: CÁC BẢNG CỦA HỆ THỐNG RAG (Sẽ tự động tạo mới)
# ==========================================

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, index=True)
    source_url = Column(String)
    
    # TRỎ VỀ ARTISTS: Dùng Uuid
    owner_id = Column(Uuid(as_uuid=True), ForeignKey("artists.id"), nullable=True) 
    created_at = Column(DateTime(timezone=True), default=get_vn_time)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"))
    chunk_text = Column(Text, nullable=False) 
    page_number = Column(Integer, nullable=True) 
    chunk_index = Column(Integer) 
    
class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # TRỎ VỀ USER CHÍNH THỨC: Dùng String + ForeignKey BẮT BUỘC
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False) 
    
    session_id = Column(String, default=lambda: str(uuid.uuid4()), index=True) 
    
    # TRỎ VỀ ARTISTS: Dùng Uuid
    artisan_id = Column(Uuid(as_uuid=True), ForeignKey("artists.id"), nullable=False) 
    
    user_query = Column(Text, nullable=False) 
    retrieved_context = Column(JSON, nullable=True) 
    ai_initial_response = Column(Text, nullable=False) 
    created_at = Column(DateTime(timezone=True), default=get_vn_time)

class GlobalUnansweredQuestion(Base):
    __tablename__ = "global_unanswered_questions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # TRỎ VỀ USER CHÍNH THỨC
    user_id = Column(String, ForeignKey("users.id"), index=True, nullable=False)
    
    user_query = Column(Text)
    session_id = Column(String)
    is_processed_by_ai_b = Column(Boolean, default=False) 
    created_at = Column(DateTime(timezone=True), default=get_vn_time)

class InterviewQueue(Base):
    __tablename__ = "interview_queue"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # TRỎ VỀ ARTISTS: Dùng Uuid
    artisan_id = Column(Uuid(as_uuid=True), ForeignKey("artists.id")) 
    question_id = Column(String, ForeignKey("global_unanswered_questions.id"))
    
    ai_b_prompt = Column(Text) 
    status = Column(String, default="pending") 
    created_at = Column(DateTime(timezone=True), default=get_vn_time)

class ArtisanAnswer(Base):
    __tablename__ = "artisan_answers"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    interview_id = Column(String, ForeignKey("interview_queue.id"))
    
    # TRỎ VỀ ARTISTS: Dùng Uuid
    artisan_id = Column(Uuid(as_uuid=True), ForeignKey("artists.id"))
    
    answer_text = Column(Text) 
    created_at = Column(DateTime(timezone=True), default=get_vn_time)

class PreDraftedQuestion(Base):
    __tablename__ = "pre_drafted_questions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    raw_topic = Column(Text, nullable=False) 
    is_used = Column(Boolean, default=False) 
    created_at = Column(DateTime(timezone=True), default=get_vn_time)