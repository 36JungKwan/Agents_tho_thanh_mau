import os
import glob
import gc
from sqlalchemy.orm import Session
from database import SessionLocal
import models
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv

# LlamaIndex & Cloud Providers
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import SentenceSplitter
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Import Gemini & Qdrant
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

load_dotenv()

# ---------------------------------------------------------
# 1. CẤU HÌNH AI & VECTOR DB (OPENAI + QDRANT CLOUD)
# ---------------------------------------------------------
print("Đang kết nối với Qdrant Cloud và Gemini...")

# 1.1 Cấu hình API Key cho toàn bộ SDK của Google
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Khởi tạo Embedding model của Gemini
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001", 
    api_key=GEMINI_API_KEY,
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

    # 2. Cấu hình OCR chạy 1 luồng
    pipeline_options = PdfPipelineOptions(do_ocr=True, num_threads=1) 
    reader = DoclingReader(pipeline_options=pipeline_options)
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # 3. ĐỌC FILE PDF GỐC ĐỂ CHIA NHỎ
    pdf_reader = PdfReader(file_path)
    total_pages = len(pdf_reader.pages)
    chunk_size = 5 # Xử lý 5 trang 1 lần (Máy yếu có thể giảm xuống 3)

    print(f"Sách có tổng cộng {total_pages} trang. Bắt đầu tiến trình OCR an toàn...")

    chunk_index_counter = 0

    # Vòng lặp cắt file PDF
    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)
        temp_filename = f"./temp_ocr_{start_page}_{end_page}.pdf"
        
        # Cắt và lưu 5 trang ra một file tạm
        pdf_writer = PdfWriter()
        for i in range(start_page, end_page):
            pdf_writer.add_page(pdf_reader.pages[i])
        
        with open(temp_filename, "wb") as f:
            pdf_writer.write(f)
            
        print(f"  -> Đang quét OCR từ trang {start_page + 1} đến {end_page}...")
        
        try:
            # Cho Docling đọc file tạm (rất nhẹ, không bao giờ sập)
            llama_docs = reader.load_data(file_path=temp_filename)
            nodes = parser.get_nodes_from_documents(llama_docs)
            
            for node in nodes:
                # Tính toán lại số trang gốc cho chuẩn xác
                page_str = node.metadata.get("page_label") or node.metadata.get("page_number")
                page_num = int(page_str) if page_str and str(page_str).isdigit() else 1
                real_page_number = start_page + page_num
                
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
                
                # Gắn Metadata để LlamaIndex tìm kiếm
                node.metadata["pg_chunk_id"] = new_chunk.id
                node.metadata["document_title"] = title
                node.metadata["page_number"] = real_page_number
                node.metadata["owner"] = "all"
                
                chunk_index_counter += 1

            # Đẩy cục Vector 5 trang này vào DB ngay lập tức
            if nodes:
                index.insert_nodes(nodes)
                
        except Exception as e:
            print(f"Lỗi ở đoạn {start_page + 1}-{end_page}: {str(e)}")
        finally:
            # 4. XÓA FILE TẠM VÀ ÉP PYTHON XẢ RÁC TRONG RAM
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            gc.collect() # chống tràn RAM

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

    print(f"Tìm thấy {len(pdf_files)} cuốn sách. Bắt đầu pipeline xử lý hàng loạt...\n")
    
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
    from database import engine
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