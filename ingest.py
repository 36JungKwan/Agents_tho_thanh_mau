import os
import glob
import gc
from sqlalchemy.orm import Session
from database import SessionLocal, engine 
import models
from PIL import Image
import fitz
import io
import uuid
from dotenv import load_dotenv

# LlamaIndex & Cloud Providers
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter

# Import Gemini & Qdrant
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# API Key Rotation Manager
from api_key_manager import key_manager

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

load_dotenv()

# ---------------------------------------------------------
# 1. CẤU HÌNH AI & VECTOR DB (OPENAI + QDRANT CLOUD)
# ---------------------------------------------------------
print("Đang kết nối với Qdrant Cloud và Gemini...")

# 1.1 Khởi tạo Embedding model với API key xoay vòng
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001", 
    api_key=key_manager.get_next_key(),
    text_task_type="RETRIEVAL_DOCUMENT"
)

# 1.2 Kết nối Qdrant Cloud
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
vector_store = QdrantVectorStore(client=client, collection_name="thomau_collection")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Tạo Index gốc để quản lý việc nhúng Vector
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)

# Khởi tạo bộ cắt chữ chuẩn
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# ---------------------------------------------------------
# 2. HÀM XỬ LÝ CHÍNH (CHẠY TRÊN MÁY TÍNH CÁ NHÂN)
# ---------------------------------------------------------
def ingest_pdf(file_path: str, title: str):
    db: Session = SessionLocal()
    
    existing_doc = db.query(models.Document).filter(models.Document.title == title).first()
    if existing_doc:
        print(f"Sách '{title}' đã tồn tại trong Database. Bỏ qua nạp lại...")
        db.close()
        return

    print(f"\n--- Đang xử lý sách: {title} ---")
    
    # 1. Tạo bản ghi Document gốc trong DB
    new_doc = models.Document(title=title, source_url=file_path, owner_id=None)
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    # Mở PDF bằng PyMuPDF
    pdf_document = fitz.open(file_path)
    total_pages = len(pdf_document)

    print(f"Sách có tổng cộng {total_pages} trang. Bắt đầu tiến trình OCR an toàn...")

    chunk_index_counter = 0

    for page_idx in range(total_pages):
        real_page_number = page_idx + 1
        page = pdf_document[page_idx]
        
        # 1. THỬ LẤY TEXT TRỰC TIẾP TRƯỚC (Rất nhanh)
        text = page.get_text("text").strip()
        
        # 2. ĐÁNH GIÁ CHẤT LƯỢNG TEXT (Bắt lỗi "Bi ế n đổ i")
        is_garbage_text = False
        if len(text) > 0:
            space_ratio = text.count(' ') / len(text)
            if space_ratio > 0.25:
                is_garbage_text = True
                
        if len(text) < 50 or is_garbage_text:
            print(f"Phát hiện trang {real_page_number} là ảnh hoặc rác. Gọi Gemini Flash đọc ảnh...")
            try:
                # Trích xuất trang thành hình ảnh độ phân giải cao
                pix = page.get_pixmap(dpi=200)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Nhờ Gemini làm đôi mắt để đọc ảnh (OCR Chuẩn xác 100%)
                prompt = "Hãy trích xuất chính xác toàn bộ văn bản tiếng Việt từ bức ảnh này. Giữ nguyên định dạng đoạn văn, không giải thích gì thêm. Chỉ in ra văn bản."
                response = key_manager.generate_with_retry(
                    model='gemini-2.5-flash',
                    contents=[img, prompt]
                )
                if response and response.text:
                    text = response.text.strip()
                else:
                    text = "" # Cho text rỗng để vòng lặp tự động bỏ qua trang này
                    print(f"Cảnh báo: AI trả về rỗng ở trang {real_page_number} (Có thể là trang trắng hoặc bị bộ lọc an toàn chặn).")
                    
            except Exception as e:
                print(f"Lỗi khi nhờ AI đọc ảnh trang {real_page_number}: {str(e)}")
                continue 
        else:
            print(f"Nạp trang {real_page_number} (Văn bản sạch, cực nhanh)...")

        if not text:
            continue
        
        try:
            # Cho Docling đọc file tạm (rất nhẹ, không bao giờ sập)
            llama_doc = LlamaDocument(text=text, metadata={"page_number": real_page_number})
            nodes = parser.get_nodes_from_documents([llama_doc])
            
            for node in nodes:
                # Lưu Text vào PostgreSQL
                new_chunk = models.DocumentChunk(
                    document_id=new_doc.id,
                    chunk_text=node.text,
                    chunk_index=chunk_index_counter,
                    page_number=real_page_number
                )
                db.add(new_chunk)
                db.commit()
                db.refresh(new_chunk)

                # Nếu chạy script 100 lần, ID này vẫn giữ nguyên -> Qdrant sẽ chỉ Ghi đè (Upsert)
                raw_id_string = f"doc_{title}_page_{real_page_number}_chunk_{chunk_index_counter}"
                node.id_ = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id_string))
                
                # Gắn Metadata để LlamaIndex tìm kiếm
                node.metadata["pg_chunk_id"] = new_chunk.id
                node.metadata["document_title"] = title
                node.metadata["page_number"] = real_page_number
                node.metadata["owner"] = "all"
                
                chunk_index_counter += 1

            if nodes:
                index.insert_nodes(nodes)
                
        except Exception as e:
            print(f"Lỗi ở trang {real_page_number}: {str(e)}")
        finally:
            gc.collect() # chống tràn RAM

    pdf_document.close()
    print(f"HOÀN TẤT! Đã nạp xong toàn bộ '{title}'.")
    db.close()

# ---------------------------------------------------------
# 3. HÀM QUÉT VÀ XỬ LÝ HÀNG LOẠT (BATCH PROCESSING)
# ---------------------------------------------------------
def process_all_pdfs_in_folder(folder_path: str):
    # Tìm tất cả các file có đuôi .pdf trong thư mục
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print(f"Không tìm thấy file PDF nào trong thư mục '{folder_path}'")
        return

    print(f"Tìm thấy {len(pdf_files)} cuốn sách. Bắt đầu đẩy lên Cloud...\n")
    
    for file_path in pdf_files:
        # Lấy tên file làm tiêu đề sách (ví dụ: "Đạo_Mẫu.pdf" -> "Đạo_Mẫu")
        title = os.path.basename(file_path).replace(".pdf", "")
        
        try:
            # Gọi hàm ingest_pdf đã viết ở trên cho từng file
            ingest_pdf(file_path=file_path, title=title)
            print("-" * 50)
        except Exception as e:
            # Bắt lỗi để nếu 1 cuốn sách bị lỗi format, hệ thống vẫn chạy tiếp cuốn sau
            print(f"Lỗi khi xử lý sách '{title}': {str(e)}")
            print("-" * 50)

# ---------------------------------------------------------
# 4. CHẠY SCRIPT
# ---------------------------------------------------------
if __name__ == "__main__":
    models.Base.metadata.create_all(bind=engine)

    # Tên thư mục chứa sách
    data_folder = "./data_sach"
    
    # Nếu thư mục chưa tồn tại, tự động tạo mới
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Đã tạo thư mục '{data_folder}'.")
        print("Vui lòng copy toàn bộ các file PDF sách vào thư mục này và chạy lại lệnh.")
    else:
        # Nếu đã có thư mục, tiến hành quét và nạp dữ liệu
        process_all_pdfs_in_folder(data_folder)
        print("QUÁ TRÌNH NẠP DỮ LIỆU ĐÃ HOÀN TẤT TOÀN BỘ!")