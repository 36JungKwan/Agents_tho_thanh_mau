from database import SessionLocal
import models
import os

def import_pre_drafted_questions(file_path="questions.txt"):
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file {file_path}")
        return

    db = SessionLocal()
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            
        count = 0
        for line in lines:
            topic = line.strip()
            if topic: # Bỏ qua dòng trống
                new_q = models.PreDraftedQuestion(raw_topic=topic)
                db.add(new_q)
                count += 1
                
        db.commit()
        print(f"Đã nạp thành công {count} chủ đề câu hỏi vào Ngân hàng kịch bản!")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # Tự động tạo bảng nếu bạn chưa chạy main.py
    from database import engine
    models.Base.metadata.create_all(bind=engine)
    
    import_pre_drafted_questions()