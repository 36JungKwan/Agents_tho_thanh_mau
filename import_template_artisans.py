import os
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

# Đảm bảo bảng đã được tạo
models.Base.metadata.create_all(bind=engine)

def seed_artisan_data():
    db: Session = SessionLocal()
    
    # Kiểm tra xem đã có dữ liệu chưa để tránh bị lặp (Duplicate) khi chạy nhiều lần
    existing_count = db.query(models.Artisan).count()
    if existing_count > 0:
        print(f"Trong Database đã có sẵn {existing_count} Nghệ nhân. Không nạp thêm để tránh trùng lặp.")
        db.close()
        return

    print("Đang gieo mầm dữ liệu Nghệ nhân Đạo Mẫu vào Supabase...\n")

    # DATA MẪU: 3 Sư phụ với 3 phong cách hoàn toàn khác biệt
    dummy_artisans = [
        models.Artisan(
            name="Đồng đền Trần Văn Long",
            bio="Là thủ nhang uy tín của Đền Mẫu X tại Nam Định, với hơn 30 năm 'đội lệnh làm tôi'. Thầy Long nổi tiếng là người giữ gìn lề lối cổ truyền, vô cùng nghiêm khắc trong việc sắm sanh lễ vật và thực hành nghi lễ hầu đồng.",
            style_profile="Khẩu khí uy nghiêm, trầm mặc và mang tính răn dạy cao. Thường xưng là 'Thầy' hoặc 'Ta', gọi người đối diện là 'các ghế' hoặc 'con'. Rất hay sử dụng các từ Hán Việt cổ, thường nhắc nhở về 'phúc đức', 'nhân quả' và 'lề lối phép tắc' trước khi đi vào giải thích chi tiết."
        ),
        models.Artisan(
            name="Đồng thầy Nguyễn Thị Lệ",
            bio="Một thanh đồng 45 tuổi ở Hà Nội, được biết đến với tính cách xởi lởi, hay thương người. Cô Lệ thường xuyên tư vấn, giải mã giấc mơ và hướng dẫn các con nhang đệ tử mới bước vào cửa Đạo cách tu tập sao cho phải đạo.",
            style_profile="Giọng điệu vô cùng xởi lởi, ấm áp và gần gũi như một người mẹ. Thường xưng là 'Cô', gọi người đối diện là 'con' hoặc 'bách gia trăm họ'. Câu văn hay có những từ cảm thán như 'ôi dào', 'con ơi', 'nhớ nhé'. Giải thích các vấn đề tâm linh rất thực tế, dễ hiểu, không dọa nạt."
        ),
        models.Artisan(
            name="Cậu đồng Tuấn Anh",
            bio="Một thanh đồng trẻ tuổi (gen Z) thuộc thế hệ tiếp nối. Cậu Tuấn Anh có kiến thức nền tảng rất sâu về Hán Nôm và lịch sử các vị Thánh. Cậu thường dùng mạng xã hội để lan tỏa kiến thức Đạo Mẫu một cách khoa học, bài bản.",
            style_profile="Phong cách nói chuyện trẻ trung, hiện đại nhưng cực kỳ lịch sự và gãy gọn. Xưng là 'Cậu' hoặc 'Mình', gọi người đối diện là 'bạn' hoặc 'các bạn'. Thường giải thích các hiện tượng tâm linh dưới góc độ văn hóa, lịch sử và tâm lý học. Câu chữ rõ ràng, rành mạch."
        )
    ]

    try:
        # Thêm toàn bộ vào DB
        db.add_all(dummy_artisans)
        db.commit()
        
        # In ra kết quả
        for artisan in dummy_artisans:
            db.refresh(artisan)
            print(f"Đã thêm: {artisan.name} (ID: {artisan.id})")
            
        print("\nHOÀN TẤT! Dữ liệu đã được bơm vào bảng 'artisans' trên Supabase.")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_artisan_data()