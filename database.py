from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from models import Base
import urllib.parse

# Tải cấu hình từ file .env
load_dotenv()

# Lấy từng thành phần ra
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

safe_password = urllib.parse.quote_plus(db_password)

# Thay bằng thông tin Database PostgreSQL của bạn
SQLALCHEMY_DATABASE_URL = f"postgresql://{db_user}:{safe_password}@{db_host}:{db_port}/{db_name}"

# Khởi tạo Engine (Bộ máy kết nối)
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Tạo Session (Phiên làm việc để truy vấn)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tự động tạo bảng nếu chưa có
Base.metadata.create_all(bind=engine)