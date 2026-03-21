from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import declarative_base, relationship
import datetime
from sqlalchemy.sql import func

Base = declarative_base()

# ==========================================
# 1. BẢNG NGHỆ NHÂN (Danh sách các Sư phụ)
# ==========================================
class Artisan(Base):
    __tablename__ = "artisans"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    bio = Column(Text) # Giới thiệu ngắn, dùng để mớm tính cách cho AI A và AI C

# ==========================================
# 2. BẢNG TÀI LIỆU & ĐOẠN VĂN (Kiến thức)
# ==========================================
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    source_url = Column(String)
    
    # ĐIỂM SÁNG TRONG KIẾN TRÚC:
    # Nếu owner_id = NULL -> Sách phổ thông, ai cũng được học.
    # Nếu owner_id = ID Nghệ nhân -> Tài liệu mật, chỉ AI A của người đó được truy cập.
    owner_id = Column(Integer, ForeignKey("artisans.id"), nullable=True) 
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    chunk_text = Column(Text, nullable=False) # Nội dung đoạn text
    page_number = Column(Integer, nullable=True) # Nằm ở trang nào trong PDF
    chunk_index = Column(Integer) # Thứ tự của chunk trong sách
    
# ==========================================
# 3. BẢNG LOG NGƯỜI DÙNG CHAT (Tương tác với AI A)
# ==========================================
class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True) # ID phiên chat của user
    artisan_id = Column(Integer, ForeignKey("artisans.id")) # Người dùng đang chat với Bản sao của Nghệ nhân nào?
    user_query = Column(Text, nullable=False) # Câu hỏi: "Hầu đồng là gì?"
    
    # Lưu lại chính xác những chunks nào đã được RAG kéo ra làm context
    # Lưu dưới dạng JSON list các DocumentChunk IDs hoặc text trực tiếp
    retrieved_context = Column(JSON, nullable=True) 
    
    ai_initial_response = Column(Text, nullable=False) # Câu trả lời gốc của Claude
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# ==========================================
# 4. BỂ CÂU HỎI CHUNG (Nơi AI A ném câu hỏi khó vào)
# ==========================================
class GlobalUnansweredQuestion(Base):
    __tablename__ = "global_unanswered_questions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_query = Column(Text)
    session_id = Column(String)
    
    # Biến cờ: Đánh dấu xem con AI B (Tổng biên tập) đã quét qua câu này chưa
    is_processed_by_ai_b = Column(Boolean, default=False) 
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# ==========================================
# 5. HỘP THƯ PHỎNG VẤN (Nơi AI C lấy bài đi hỏi Sư phụ)
# ==========================================
class InterviewQueue(Base):
    __tablename__ = "interview_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    artisan_id = Column(Integer, ForeignKey("artisans.id")) # Gửi cho Nghệ nhân nào?
    question_id = Column(Integer, ForeignKey("global_unanswered_questions.id"))
    
    # Câu hỏi đã được AI B viết lại cho mềm mại, kính trọng (Ví dụ: "Dạ thưa thầy...")
    ai_b_prompt = Column(Text) 
    
    # Trạng thái: pending (chờ hỏi), answered (đã trả lời), skipped (thầy từ chối trả lời)
    status = Column(String, default="pending") 
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# ==========================================
# 6. DI SẢN KIẾN THỨC (Câu trả lời của Nghệ nhân)
# ==========================================
class ArtisanAnswer(Base):
    __tablename__ = "artisan_answers"
    
    id = Column(Integer, primary_key=True, index=True)
    interview_id = Column(Integer, ForeignKey("interview_queue.id"))
    artisan_id = Column(Integer, ForeignKey("artisans.id"))
    
    # Text mà App (AI C) nhận được (Gõ phím hoặc Ghi âm chuyển thành Text)
    answer_text = Column(Text) 
    created_at = Column(DateTime, default=datetime.datetime.utcnow)