from fastapi import FastAPI, Depends, BackgroundTasks, File, UploadFile, Form, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import os
import shutil
import uuid 
from fastapi.responses import StreamingResponse
from qdrant_client import models as qdrant_models

# LlamaIndex & Cloud Vector DB Imports
from llama_index.core import Settings, VectorStoreIndex, StorageContext, Document as LlamaDocument
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator, FilterCondition
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import google.genai as genai
from google.genai import types as genai_types
from google.genai.types import Content, Part

# Lightweight PDF Reader cho Serverless
import io
from PIL import Image
import fitz
from dotenv import load_dotenv

import models
from database import SessionLocal, engine

load_dotenv()

# Tự động tạo bảng nếu chưa có
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Thờ Mẫu RAG API")

# Cho phép Frontend gọi API (CORS)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# CẤU HÌNH AI (Gemini & LlamaIndex) - SỬ DỤNG KEY ROTATION
# ---------------------------------------------------------
from api_key_manager import key_manager

# Khởi tạo Embedding model với API key xoay vòng
Settings.embed_model = key_manager.get_embed_model()

# Kết nối Qdrant Cloud
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

try:
    client.create_payload_index(
        collection_name="thomau_collection",
        field_name="owner",
        field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
    )
except Exception as e:
    # Nếu Index đã được tạo từ lần chạy trước, nó sẽ báo lỗi, ta cứ thản nhiên bỏ qua
    pass

vector_store = QdrantVectorStore(client=client, collection_name="thomau_collection")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Tạo Index gốc để quản lý việc nhúng Vector
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)

# ---------------------------------------------------------
# CẤU TRÚC REQUEST
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str     # Nhận diện chính xác Người dùng (Tài khoản/SĐT/Email)
    session_id: str  # ID của phiên chat hiện tại
    artisan_id: str  # Đang chat với Nghệ nhân nào
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
def save_background_logs(db: Session, user_id: str, session_id: str, artisan_id: int, original_query: str, search_query: str, ai_answer: str, context_metadata: list):
    # 1. Lưu vào ChatLogs (Vẫn lưu câu gốc để giữ nguyên lịch sử chat tự nhiên)
    new_log = models.ChatLog(
        user_id=user_id,
        session_id=session_id,
        artisan_id=artisan_id,
        user_query=original_query, 
        ai_initial_response=ai_answer,
        retrieved_context=context_metadata
    )
    db.add(new_log)

    # 2. KIỂM TRA TỪ KHÓA BẮT LỖI
    # Nếu AI A thú nhận là không biết, lập tức quăng câu hỏi vào Bể chung
    if "chưa có tài liệu ghi chép" in ai_answer.lower():
        # LƯU Ý: Dùng 'search_query' (câu đã viết lại đủ ngữ cảnh) ném vào Bể chung cho AI B
        unanswered = models.GlobalUnansweredQuestion(
            user_id=user_id,
            user_query=search_query,
            session_id=session_id
        )
        db.add(unanswered)
        print(f"[LOG] Đã lưu câu hỏi khó từ user {user_id} vào Bể chung: '{search_query}'")
        
    db.commit()

# ---------------------------------------------------------
# 4. API CHÍNH: BẢN SAO SỐ AI A TRẢ LỜI NGƯỜI DÙNG (CÓ TRÍ NHỚ)
# ---------------------------------------------------------
@app.post("/api/chat")
async def chat_with_artisan_twin(request: ChatRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    original_query = request.user_query
    session_id = request.session_id
    user_id = request.user_id
    artisan_id = request.artisan_id
    
    # ==========================================
    # BƯỚC 1: LẤY LỊCH SỬ TRÒ CHUYỆN (TRÍ NHỚ)
    # ==========================================
    # Chỉ lấy lịch sử của user này với ĐÚNG vị Nghệ nhân này
    history_logs = db.query(models.ChatLog).filter(
        models.ChatLog.session_id == user_id,
        models.ChatLog.session_id == session_id,
        models.ChatLog.artisan_id == artisan_id
    ).order_by(models.ChatLog.created_at.desc()).limit(4).all()
    
    history_logs = history_logs[::-1] # Đảo ngược từ cũ đến mới
    
    history_text = ""
    messages_for_gemini = []
    
    for log in history_logs:
        history_text += f"User: {log.user_query}\nAI: {log.ai_initial_response}\n"
        # Bọc chữ bằng Part.from_text, sau đó bọc thành Content
        messages_for_gemini.append(Content(role="user", parts=[Part.from_text(text=log.user_query)]))
        messages_for_gemini.append(Content(role="model", parts=[Part.from_text(text=log.ai_initial_response)]))

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
        
        # Gọi Gemini 1 nhịp nhanh để rewrite (Có xoay key tự động)
        rewrite_response = key_manager.generate_with_retry(
            model='gemini-2.5-flash',
            contents=rewrite_prompt
        )
        if rewrite_response and rewrite_response.text:
            search_query = rewrite_response.text.strip()
            print(f"[LOG] Câu hỏi gốc: '{original_query}' -> Viết lại thành: '{search_query}'")

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

    retriever = index.as_retriever(similarity_top_k=5, filters=filters)
    
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
    # BƯỚC 5: GỌI Gemini TRẢ LỜI CÓ GÀI BẪY
    # ==========================================
    artisan = db.query(models.Artisan).filter(models.Artisan.id == request.artisan_id).first()
    
    # Đề phòng trường hợp Frontend truyền sai ID
    if not artisan:
        raise HTTPException(status_code=404, detail="Không tìm thấy Nghệ nhân này trong hệ thống!")
        
    artisan_name = artisan.name
    artisan_bio = artisan.bio if artisan.bio else "Một bậc thầy am hiểu sâu sắc về Tín ngưỡng Thờ Mẫu."
    core_persona = artisan.style_profile if artisan.style_profile else "Nói chuyện từ tốn, uy nghiêm nhưng gần gũi."

    system_instruction = (
        f"Bạn là {artisan_name}. {artisan_bio}\n\n"
        "Bạn đang đóng vai là Bản sao số (Digital Twin) của chính mình để truyền dạy và giải đáp thắc mắc về Đạo Mẫu. "
        "ĐÂY LÀ HƯỚNG DẪN BẮT BUỘC VỀ PHONG CÁCH GIAO TIẾP CỦA BẠN (Phải tuân thủ tuyệt đối):\n"
        f"> {core_persona}\n\n"
        "Hãy dùng thông tin tham khảo dưới đây để tiếp tục cuộc trò chuyện. Hãy trả lời thật tự nhiên như đang nói chuyện trực tiếp.\n\n"
        "ĐIỀU KIỆN TỐI QUAN TRỌNG: Nếu thông tin tham khảo KHÔNG chứa câu trả lời và bạn không biết chắc chắn, "
        "bạn TUYỆT ĐỐI KHÔNG ĐƯỢC BỊA ĐẶT. Bạn phải nói câu có chứa cụm từ sau: "
        "'chưa có tài liệu ghi chép' để hẹn người dùng trả lời sau.\n\n"
        f"--- THÔNG TIN THAM KHẢO ---\n{joined_context}\n--------------------------"
    )
    
    # Đẩy câu hỏi mới nhất vào mảng
    messages_for_gemini.append(Content(role="user", parts=[Part.from_text(text=original_query)]))
    
    # Bật tính năng truyền phát trực tiếp (Stream) - CÓ XOAY KEY TỰ ĐỘNG
    response_stream = key_manager.generate_with_retry(
        model="gemini-2.5-pro",
        contents=messages_for_gemini,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=4096,
            temperature=0.3 
        ),
        stream=True
    )
    
    # Hàm Generator xử lý Stream và Lưu Background
    async def stream_generator():
        full_ai_answer = ""
        try:
            for chunk in response_stream:
                if chunk.text:
                    full_ai_answer += chunk.text
                    yield chunk.text
                    
            # Nếu thành công trọn vẹn thì lưu log
            background_tasks.add_task(
                save_background_logs, db, user_id, session_id, artisan_id,
                original_query, search_query, full_ai_answer, saved_context_metadata
            )
        except Exception as e:
            # In lỗi thật ra terminal để debug
            print(f"[STREAM ERROR] Lỗi khi stream từ Gemini: {type(e).__name__}: {str(e)}")
            # Nếu Google sập giữa chừng, Sư phụ sẽ nói câu này:
            error_msg = "\n\n(Dạ thưa, hiện tại tâm linh đang nhiễu loạn, Sư phụ cần nghỉ ngơi một lát. Xin con quay lại sau ít phút nhé!)"
            yield error_msg
    
    return StreamingResponse(
        stream_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Tắt buffer cho Nginx nếu dùng reverse proxy
            "Connection": "keep-alive",
        }
    )

# =========================================================
# 5. API CHO AI C (ĐỆ TỬ ẢO TRÊN MOBILE APP CỦA NGHỆ NHÂN)
# =========================================================

@app.get("/api/artisan/{artisan_id}/questions")
async def get_pending_questions(artisan_id: str, db: Session = Depends(get_db)):
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
    artisan_id: str,
    background_tasks: BackgroundTasks,
    interview_id: str = Form(...),
    answer_text: str | None = Form(None),
    upload_file: UploadFile | None = File(None),
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
    # TRƯỜNG HỢP 1: SƯ PHỤ TẢI FILE PDF (Chốt luôn, không hỏi vặn vẹo)
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
        task.status = "answered"
        db.commit()
        return {"action": "completed", 
                "message": "Dạ con cám ơn Sư phụ đã chia sẻ tài liệu quý báu ạ. Lời dạy của Sư phụ đã được con ghi nhớ."
        }
    
    # ---------------------------------------------------------
    # TRƯỜNG HỢP 2: SƯ PHỤ GÕ CHỮ (AI C đánh giá và trò chuyện)
    # ---------------------------------------------------------
    if answer_text:
        # Gọi Gemini (AI C) vào đánh giá chất lượng câu trả lời
        eval_prompt = (
            f"Bạn là một người phỏng vấn. Bạn đã hỏi Sư phụ câu này: '{task.ai_b_prompt}'\n"
            f"Sư phụ trả lời: '{answer_text}'\n\n"
            "Nhiệm vụ: Hãy đánh giá xem câu trả lời này đã đủ chi tiết chưa. "
            "Nếu quá ngắn gọn, lấp lửng hoặc chưa rõ ràng, hãy đóng vai Đệ tử ngoan ngoãn, đặt tiếp 1 câu hỏi lễ phép để khai thác sâu hơn. "
            "Nếu câu trả lời đã đủ chi tiết và trọn vẹn, CHỈ CẦN IN RA DUY NHẤT CHỮ 'OK'."
        )
        eval_response = key_manager.generate_with_retry(
            model='gemini-2.5-flash',
            contents=eval_prompt
        )
        eval_result = eval_response.text.strip() if eval_response and eval_response.text else "OK"
        
        # LƯU DI SẢN (Dù ngắn hay dài cũng lưu lại để máy học văn phong)
        new_answer = models.ArtisanAnswer(
            interview_id=interview_id, artisan_id=artisan_id, answer_text=answer_text
        )
        db.add(new_answer)
        
        # Nếu AI C thấy chưa thỏa mãn -> Hỏi tiếp
        if eval_result != "OK":
            # Cập nhật lại câu hỏi AI B Prompt thành câu hỏi mới để App hiển thị tiếp
            task.ai_b_prompt = f"{task.ai_b_prompt}\nSư phụ: {answer_text}\nĐệ tử: {eval_result}"
            db.commit()
            
            return {
                "action": "continue_chat", 
                "message": eval_result # Bắn câu hỏi xoáy về cho App Mobile hiển thị
            }

        # Nếu AI C thấy 'OK' (Đã thỏa mãn) -> Nạp Vector và kết thúc
        full_context_text = f"Vấn đề: {task.ai_b_prompt}\nSư phụ giải đáp: {answer_text}"
        doc = LlamaDocument(
            text=full_context_text,
            metadata={"document_title": f"Hồ sơ vấn đáp của Sư phụ {artisan_id}", "owner": f"artisan_{artisan_id}"}
        )
        index.insert(doc) 
        
        task.status = "answered"
        db.commit()
        
        return {
            "action": "completed", 
            "message": "Dạ con đã ghi chép lại đầy đủ. Cám ơn Sư phụ đã chỉ dạy ạ."
        }


# ---------------------------------------------------------
# HÀM CHẠY NGẦM: BÓC TÁCH FILE PDF RIÊNG CỦA NGHỆ NHÂN
# ---------------------------------------------------------
def process_artisan_private_pdf(db: Session, artisan_id: int, file_path: str, filename: str):
    try:
        # 1. Sử dụng key_manager để xoay key khi OCR
        
        # 2. Mở file PDF bằng PyMuPDF
        pdf_document = fitz.open(file_path)
        llama_docs = []
        
        print(f"Bắt đầu xử lý tài liệu mật '{filename}' của Sư phụ {artisan_id}...")

        for page_idx in range(len(pdf_document)):
            real_page_number = page_idx + 1
            page = pdf_document[page_idx]
            text = page.get_text("text").strip()
            
            # Bộ lọc AI: Kiểm tra xem có phải là ảnh scan sách cúng không
            is_garbage_text = False
            if len(text) > 0:
                space_ratio = text.count(' ') / len(text)
                if space_ratio > 0.25:
                    is_garbage_text = True
                    
            if len(text) < 50 or is_garbage_text:
                print(f"[Private] Trang {real_page_number} là ảnh. Đang dùng Gemini Flash đọc...")
                try:
                    pix = page.get_pixmap(dpi=200)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    prompt = "Hãy trích xuất chính xác toàn bộ văn bản tiếng Việt từ bức ảnh này. Giữ nguyên định dạng đoạn văn, không giải thích gì thêm. Chỉ in ra văn bản."
                    response = key_manager.generate_with_retry(
                        model='gemini-2.5-flash',
                        contents=[img, prompt]
                    )
                    text = response.text.strip()
                except Exception as e:
                    print(f"[Private] Lỗi AI đọc ảnh trang {real_page_number}: {str(e)}")
                    continue
            else:
                print(f"[Private] Đã đọc xong văn bản trang {real_page_number}.")

            if text and text.strip():
                llama_docs.append(LlamaDocument(
                    text=text, 
                    metadata={"page_number": real_page_number}
                ))
                
        pdf_document.close()

        if not llama_docs:
            print("Tài liệu mật bị trống hoặc lỗi. Không có gì để nạp.")
            return
        
        # 3. Cắt chunk
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = parser.get_nodes_from_documents(llama_docs)
        
        # 4. Lưu vào Supabase PostgreSQL
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
            page_str = node.metadata.get("page_number") or node.metadata.get("page_label")
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

            # Nếu chạy script 100 lần, ID này vẫn giữ nguyên -> Qdrant sẽ chỉ Ghi đè (Upsert)
            raw_id_string = f"doc_{filename}_{artisan_id}_page_{real_page_number}_chunk_{i}"
            node.id_ = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id_string))
            
            # GẮN NHÃN BẢN QUYỀN VÀO TỪNG NODE TRONG QDRANT
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
