import os
from sqlalchemy.orm import Session
from database import SessionLocal
import models
import glob

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from docling.datamodel.pipeline_options import PdfPipelineOptions
import chromadb

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ---------------------------------------------------------
# 1. CẤU HÌNH AI & VECTOR DB
# ---------------------------------------------------------

# Sử dụng model tiếng Việt miễn phí, cực tốt cho embedding văn bản
print("Đang tải Embedding Model...")
Settings.embed_model = HuggingFaceEmbedding(model_name="keepitreal/vietnamese-sbert")

# Khởi tạo ChromaDB lưu trên ổ cứng (thư mục ./chroma_db)
db_chroma = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db_chroma.get_or_create_collection("thomau_collection")

# Kết nối ChromaDB với LlamaIndex
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ---------------------------------------------------------
# 2. HÀM XỬ LÝ CHÍNH
# ---------------------------------------------------------

def ingest_pdf(file_path: str, title: str):
    db: Session = SessionLocal()
    
    # 1. Kiểm tra sách trùng lặp
    existing_doc = db.query(models.Document).filter(models.Document.title == title).first()
    if existing_doc:
        print(f"⏩ Sách '{title}' đã tồn tại trong Database. Bỏ qua nạp lại...")
        db.close()
        return

    print(f"\n--- Đang xử lý sách: {title} ---")

    # 2. Cấu hình đọc PDF, chỉnh lại OCR để không bị tràn RAM
    pipeline_options = PdfPipelineOptions(
        do_ocr=True, 
        num_threads=1 # Ép nó chỉ được chạy 1 luồng duy nhất (tuần tự từng trang một)
    ) 
    reader = DoclingReader(pipeline_options=pipeline_options)
    
    llama_docs = reader.load_data(file_path=file_path)
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(llama_docs)
    
    # 3. Lưu thông tin sách vào PostgreSQL (owner_id = None nghĩa là Sách chung)
    new_doc = models.Document(
        title=title, 
        source_url=file_path,
        owner_id=None # <-- ĐIỂM QUAN TRỌNG: Sách phổ thông
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    
    print(f"Đã chia thành {len(nodes)} đoạn văn nhỏ. Đang lưu vào Database...")
    
    # 4. Lưu từng Chunk vào Postgres & ChromaDB
    for i, node in enumerate(nodes):

        page_str = node.metadata.get("page_label") or node.metadata.get("page_number")
        page_num = int(page_str) if page_str and str(page_str).isdigit() else None

        # Lưu Text gốc vào PostgreSQL
        new_chunk = models.DocumentChunk(
            document_id=new_doc.id,
            chunk_text=node.text,
            chunk_index=i,
            page_number=page_num
        )
        db.add(new_chunk)
        db.commit()
        db.refresh(new_chunk)
        
        # Gắn thẻ metadata vào ChromaDB để AI A phục vụ việc LỌC
        node.metadata["pg_chunk_id"] = new_chunk.id
        node.metadata["document_title"] = title
        node.metadata["page_number"] = page_num
        node.metadata["owner"] = "all" # DÁN NHÃN SÁCH CHUNG VÀO VECTOR

    # Bước E: Nhúng Vector và lưu vào ChromaDB
    print("Đang tạo Vector và lưu vào ChromaDB...")
    VectorStoreIndex(nodes, storage_context=storage_context)
    
    print(f"HOÀN TẤT! Đã nạp {len(nodes)} chunks của '{title}' vào hệ thống.")
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