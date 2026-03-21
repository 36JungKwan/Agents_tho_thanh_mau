from fastapi import FastAPI, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
import os
import chromadb

# LlamaIndex Imports
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator, FilterCondition

from fastapi import File, UploadFile, Form, HTTPException
from typing import Optional
import shutil
from llama_index.core import Document as LlamaDocument
from llama_index.readers.docling import DoclingReader
from docling.datamodel.pipeline_options import PdfPipelineOptions
from llama_index.core.node_parser import SentenceSplitter

from anthropic import Anthropic
import models
from database import SessionLocal, engine
from dotenv import load_dotenv

load_dotenv()

# Tự động tạo bảng nếu chưa có
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Thờ Mẫu RAG API")

# ---------------------------------------------------------
# CẤU HÌNH AI (Claude & LlamaIndex)
# ---------------------------------------------------------
claude_client = Anthropic(api_key=os.getenv("SECRET_API_KEY"))

# Khởi tạo Embedding model (Phải giống hệt model lúc Ingest)
Settings.embed_model = HuggingFaceEmbedding(model_name="keepitreal/vietnamese-sbert")

# Kết nối vào ChromaDB đã lưu trên ổ cứng
db_chroma = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db_chroma.get_or_create_collection("thomau_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Tạo Retriever (Bộ truy xuất) để lấy top 4 đoạn văn liên quan nhất
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)

# ---------------------------------------------------------
# CẤU TRÚC REQUEST
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    artisan_id: int # Yêu cầu người dùng chọn Nghệ nhân
    user_query: str

# Dependency lấy DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------------------------------------
# 3. TASK CHẠY NGẦM: LƯU LOG & BẮT CÂU HỎI KHÓ
# ---------------------------------------------------------
def save_background_logs(db: Session, session_id: str, artisan_id: int, original_query: str, search_query: str, ai_answer: str, context_metadata: list):
    # 1. Lưu vào ChatLogs (Vẫn lưu câu gốc để giữ nguyên lịch sử chat tự nhiên)
    new_log = models.ChatLog(
        session_id=session_id,
        artisan_id=artisan_id,
        user_query=original_query, 
        ai_initial_response=ai_answer,
        retrieved_context=context_metadata
    )
    db.add(new_log)

    # 2. KIỂM TRA TỪ KHÓA BẮT LỖI
    # Nếu AI A thú nhận là không biết, lập tức quăng câu hỏi vào Bể chung
    if "tôi chưa có tài liệu ghi chép" in ai_answer.lower():
        # LƯU Ý: Dùng 'search_query' (câu đã viết lại đủ ngữ cảnh) ném vào Bể chung cho AI B
        unanswered = models.GlobalUnansweredQuestion(
            user_query=search_query,
            session_id=session_id
        )
        db.add(unanswered)
        print(f"[LOG] Đã lưu câu hỏi khó vào Bể chung: '{search_query}'")
        
    db.commit()

# ---------------------------------------------------------
# 4. API CHÍNH: BẢN SAO SỐ AI A TRẢ LỜI NGƯỜI DÙNG (CÓ TRÍ NHỚ)
# ---------------------------------------------------------
@app.post("/api/chat")
async def chat_with_artisan_twin(request: ChatRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    original_query = request.user_query
    session_id = request.session_id
    
    # ==========================================
    # BƯỚC 1: LẤY LỊCH SỬ TRÒ CHUYỆN (TRÍ NHỚ)
    # ==========================================
    # Chỉ lấy lịch sử của user này với ĐÚNG vị Nghệ nhân này
    history_logs = db.query(models.ChatLog).filter(
        models.ChatLog.session_id == request.session_id,
        models.ChatLog.artisan_id == request.artisan_id
    ).order_by(models.ChatLog.created_at.desc()).limit(4).all()
    
    history_logs = history_logs[::-1] # Đảo ngược từ cũ đến mới
    
    history_text = ""
    messages_for_claude = []
    
    for log in history_logs:
        history_text += f"User: {log.user_query}\nAI: {log.ai_initial_response}\n"
        messages_for_claude.append({"role": "user", "content": log.user_query})
        messages_for_claude.append({"role": "assistant", "content": log.ai_initial_response})

    # ---------------------------------------------------------
    # BƯỚC 2: VIẾT LẠI CÂU HỎI (NẾU CÓ LỊCH SỬ) ĐỂ TÌM KIẾM VECTOR
    # ---------------------------------------------------------
    search_query = original_query
    if history_text.strip():
        rewrite_prompt = (
            f"Dựa vào đoạn hội thoại trước đó:\n{history_text}\n"
            f"Người dùng vừa hỏi câu mới: '{original_query}'.\n"
            "Nhiệm vụ: Hãy viết lại câu hỏi mới này thành một câu hỏi độc lập, đầy đủ chủ ngữ và ngữ cảnh để dùng cho công cụ tìm kiếm tài liệu. "
            "KHÔNG trả lời câu hỏi. CHỈ in ra câu hỏi đã được viết lại."
        )
        
        # Gọi Claude 1 nhịp nhanh để rewrite (Tốn vài mili-giây)
        rewrite_response = claude_client.messages.create(
            model="claude-haiku-4-5-20251001", # Hoặc tên model bạn đang dùng chạy thành công
            max_tokens=50,
            messages=[{"role": "user", "content": rewrite_prompt}]
        )
        search_query = rewrite_response.content[0].text.strip()
        print(f"🔍 [LOG] Câu hỏi gốc: '{original_query}' -> Viết lại thành: '{search_query}'")

   # ==========================================
    # BƯỚC 3: BỘ LỌC ĐA KHÁCH THUÊ (MULTI-TENANT)
    # ==========================================
    artisan_tag = f"artisan_{request.artisan_id}"
    
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="owner", value="all", operator=FilterOperator.EQ),
            MetadataFilter(key="owner", value=artisan_tag, operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.OR
    )

    retriever = index.as_retriever(similarity_top_k=4, filters=filters)
    
    # ==========================================
    # BƯỚC 4: TÌM KIẾM NGỮ CẢNH BẰNG CÂU HỎI ĐÃ VIẾT LẠI
    # ==========================================
    retrieved_nodes = retriever.retrieve(search_query)
    
    context_texts = []
    saved_context_metadata = []
    
    for idx, node in enumerate(retrieved_nodes):
        doc_title = node.metadata.get("document_title", "Không rõ sách")
        chunk_id = node.metadata.get("pg_chunk_id", "N/A")
        page_num = node.metadata.get("page_number", "N/A")
        owner = node.metadata.get("owner", "all")
        
        text_snippet = f"[Nguồn {idx+1}: {doc_title} - Trang {page_num} - Sở hữu: {owner}] {node.text}"
        context_texts.append(text_snippet)
        
        saved_context_metadata.append({
            "pg_chunk_id": chunk_id,
            "document_title": doc_title,
            "page": page_num,
            "owner": owner,
            "text": node.text
        })
    
    joined_context = "\n\n".join(context_texts)
    
    # ==========================================
    # BƯỚC 5: GỌI CLAUDE TRẢ LỜI CÓ GÀI BẪY
    # ==========================================
    system_prompt = (
        "Bạn là Bản sao AI của một Nghệ nhân Tín ngưỡng Thờ Mẫu. "
        "Giọng điệu trang trọng, từ tốn, uy nghiêm nhưng gần gũi. Xưng 'tôi' và gọi 'bạn'. "
        "Hãy dùng thông tin tham khảo dưới đây để tiếp tục cuộc trò chuyện.\n\n"
        "ĐIỀU KIỆN TỐI QUAN TRỌNG: Nếu thông tin tham khảo KHÔNG chứa câu trả lời và bạn không biết chắc chắn, "
        "bạn TUYỆT ĐỐI KHÔNG ĐƯỢC BỊA ĐẶT. Bạn phải nói câu có chứa cụm từ sau: "
        "'tôi chưa có tài liệu ghi chép' để hẹn người dùng trả lời sau.\n\n"
        f"--- THÔNG TIN THAM KHẢO ---\n{joined_context}\n--------------------------"
    )
    
    # Thêm câu hỏi mới nhất vào mảng tin nhắn gửi cho Claude
    messages_for_claude.append({"role": "user", "content": original_query})
    
    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001", # Hoặc tên model bạn đang dùng
        max_tokens=1000,
        system=system_prompt,
        messages=messages_for_claude
    )
    
    ai_answer = response.content[0].text
    
    # ==========================================
    # BƯỚC 6: LƯU LOG & BẮT CÂU HỎI KHÓ (CHẠY NGẦM)
    # ==========================================
    background_tasks.add_task(
        save_background_logs, db,
        request.session_id, request.artisan_id,
        original_query, search_query, ai_answer,
        saved_context_metadata
    )
    
    return {
        "artisan_id_used": request.artisan_id,
        "user_query": original_query,
        "search_query_used": search_query,
        "ai_response": ai_answer,
        "sources": saved_context_metadata
    }

# =========================================================
# 5. API CHO AI C (ĐỆ TỬ ẢO TRÊN MOBILE APP CỦA NGHỆ NHÂN)
# =========================================================

@app.get("/api/artisan/{artisan_id}/questions")
async def get_pending_questions(artisan_id: int, db: Session = Depends(get_db)):
    """
    API 1: Khi Nghệ nhân mở App, AI C gọi API này để lấy danh sách câu hỏi đang chờ.
    """
    questions = db.query(models.InterviewQueue).filter(
        models.InterviewQueue.artisan_id == artisan_id,
        models.InterviewQueue.status == "pending"
    ).all()
    
    result = []
    for q in questions:
        result.append({
            "interview_id": q.id,
            "ai_b_prompt": q.ai_b_prompt, # Câu hỏi lễ phép AI B đã soạn
            "created_at": q.created_at
        })
        
    return {
        "artisan_id": artisan_id,
        "pending_tasks_count": len(result),
        "questions": result
    }

@app.post("/api/artisan/{artisan_id}/answer")
async def submit_artisan_answer(
    artisan_id: int,
    background_tasks: BackgroundTasks,
    interview_id: int = Form(...),
    answer_text: Optional[str] = Form(None),
    upload_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    API 2: Nghệ nhân nộp câu trả lời (Gõ chữ, Thu âm chuyển text, HOẶC Tải file PDF)
    """
    # 1. Kiểm tra xem câu hỏi có tồn tại không
    task = db.query(models.InterviewQueue).filter(
        models.InterviewQueue.id == interview_id,
        models.InterviewQueue.artisan_id == artisan_id
    ).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Không tìm thấy câu hỏi này trong hộp thư của Sư phụ!")

    # ---------------------------------------------------------
    # TRƯỜNG HỢP 1: NGHỆ NHÂN GÕ CHỮ (Hoặc nói vào mic chuyển thành Text)
    # ---------------------------------------------------------
    if answer_text:
        # Lưu vào PostgreSQL để làm Di sản
        new_answer = models.ArtisanAnswer(
            interview_id=interview_id,
            artisan_id=artisan_id,
            answer_text=answer_text
        )
        db.add(new_answer)
        db.commit()

        # Gộp cả Câu hỏi + Câu trả lời vào chung 1 khối Text để AI A dễ tìm kiếm sau này
        full_context_text = f"Vấn đề: {task.ai_b_prompt}\nSư phụ giảng giải: {answer_text}"

        # Biến đoạn text thành Vector và nhét thẳng vào ChromaDB
        # Gắn nhãn owner chính xác là Sư phụ này
        doc = LlamaDocument(
            text=full_context_text,
            metadata={
                "document_title": f"Hồ sơ vấn đáp của Sư phụ {artisan_id}", # Title ngắn gọn, an toàn
                "owner": f"artisan_{artisan_id}" # Đóng dấu bản quyền
            }
        )
        # Hàm insert tự động cắt chunk và gọi Embedding model
        index.insert(doc) 
        print(f"Đã nạp trọn vẹn bối cảnh (Hỏi & Đáp) của Sư phụ {artisan_id} vào Não bộ AI A.")

    # ---------------------------------------------------------
    # TRƯỜNG HỢP 2: NGHỆ NHÂN TẢI FILE PDF (Tài liệu nghiên cứu riêng)
    # ---------------------------------------------------------
    if upload_file:
        # Lưu file tạm thời lên Server
        temp_file_path = f"./temp_{upload_file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        print(f"Đã nhận tài liệu mật '{upload_file.filename}' từ Sư phụ {artisan_id}. Đang xử lý ngầm...")
        
        # ĐƯA VÀO BACKGROUND TASK (Xử lý ngầm để App của Sư phụ không bị treo khi chờ đọc PDF)
        background_tasks.add_task(
            process_artisan_private_pdf, db, artisan_id, temp_file_path, upload_file.filename
        )

    # ---------------------------------------------------------
    # CẬP NHẬT TRẠNG THÁI: ĐÃ TRẢ LỜI XONG
    # ---------------------------------------------------------
    task.status = "answered"
    db.commit()
    
    return {"message": "Dạ con cám ơn Sư phụ! Lời dạy của Sư phụ đã được con ghi nhớ và truyền lại cho người hỏi ạ."}


# ---------------------------------------------------------
# HÀM CHẠY NGẦM: BÓC TÁCH FILE PDF RIÊNG CỦA NGHỆ NHÂN
# ---------------------------------------------------------
def process_artisan_private_pdf(db: Session, artisan_id: int, file_path: str, filename: str):
    try:
        # Đọc nhanh không cần OCR để tiết kiệm RAM server
        pipeline_options = PdfPipelineOptions(do_ocr=False) 
        reader = DoclingReader(pipeline_options=pipeline_options)
        
        llama_docs = reader.load_data(file_path=file_path)
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(llama_docs)
        
        # 1. Lưu thông tin sách MẬT vào PostgreSQL
        new_doc = models.Document(
            title=filename, 
            source_url="artisan_upload",
            owner_id=artisan_id # KHÓA CHẶT: Chỉ AI A của Sư phụ này mới được đọc
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        
        # 2. Lưu từng Chunk
        for i, node in enumerate(nodes):
            page_str = node.metadata.get("page_label") or node.metadata.get("page_number")
            page_num = int(page_str) if page_str and str(page_str).isdigit() else None
            
            new_chunk = models.DocumentChunk(
                document_id=new_doc.id,
                chunk_text=node.text,
                chunk_index=i,
                page_number=page_num
            )
            db.add(new_chunk)
            db.commit()
            db.refresh(new_chunk)
            
            # GẮN NHÃN BẢN QUYỀN VÀO TỪNG NODE TRONG CHROMADB
            node.metadata["pg_chunk_id"] = new_chunk.id
            node.metadata["document_title"] = filename
            node.metadata["page_number"] = page_num
            node.metadata["owner"] = f"artisan_{artisan_id}"
            
        # 3. Đẩy Vector
        VectorStoreIndex(nodes, storage_context=storage_context)
        print(f"Xong! Đã nạp thành công tài liệu mật của Sư phụ {artisan_id} vào hệ thống.")
        
    except Exception as e:
        print(f"Lỗi khi đọc file mật của Sư phụ {artisan_id}: {str(e)}")
    finally:
        # Xóa file tạm cho nhẹ ổ cứng
        if os.path.exists(file_path):
            os.remove(file_path)