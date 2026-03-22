from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import urllib.parse
from models import Base

# Tải cấu hình từ file .env
load_dotenv()

# Lấy từng thành phần ra
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

# Mã hóa mật khẩu
safe_password = urllib.parse.quote_plus(db_password)

# Chuỗi kết nối Postgres an toàn
SQLALCHEMY_DATABASE_URL = f"postgresql://{db_user}:{safe_password}@{db_host}:{db_port}/{db_name}"

# Khởi tạo Engine (Bộ máy kết nối)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,      # BẮT BUỘC: Kiểm tra kết nối có bị Supabase/Render ngắt ngầm không trước khi dùng
    pool_recycle=300,        # Làm mới kết nối sau mỗi 5 phút (300 giây) để tránh timeout
    pool_size=5,             # Giữ tối đa 5 kết nối mở cùng lúc (phù hợp gói Free của Supabase)
    max_overflow=10          # Cho phép phình ra thêm 10 kết nối lúc cao điểm
)

# Tạo Session (Phiên làm việc để truy vấn)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tự động tạo bảng nếu chưa có
Base.metadata.create_all(bind=engine)