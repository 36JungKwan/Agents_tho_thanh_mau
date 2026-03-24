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
            
        success_count = 0
        skip_count = 0
        
        print(f"Đang quét dữ liệu từ file '{file_path}'...")
        
        for line in lines:
            topic = line.strip()
            if topic: # Bỏ qua dòng trống
                # ---------------------------------------------------
                # BƯỚC CHECK TRÙNG LẶP (DUPLICATE CHECK)
                # ---------------------------------------------------
                existing_q = db.query(models.PreDraftedQuestion).filter(
                    models.PreDraftedQuestion.raw_topic == topic
                ).first()
                
                if existing_q:
                    print(f"Bỏ qua: Kịch bản '{topic}' đã tồn tại trong hệ thống.")
                    skip_count += 1
                    continue
                # ---------------------------------------------------
                
                # Nạp mới nếu chưa có
                new_q = models.PreDraftedQuestion(raw_topic=topic)
                db.add(new_q)
                success_count += 1
                print(f"Đã xếp hàng nạp mới: '{topic}'")
                
        # Commit 1 lần duy nhất vào Supabase cho tốc độ cực nhanh
        db.commit()
        print(f"\nHOÀN TẤT NẠP NGÂN HÀNG KỊCH BẢN!")
        print(f"   - Nạp mới thành công: {success_count} câu.")
        print(f"   - Bỏ qua (Đã tồn tại): {skip_count} câu.")
        
    except Exception as e:
        print(f"\nCó lỗi xảy ra: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # Tự động tạo bảng nếu bạn chưa chạy main.py
    from database import engine
    models.Base.metadata.create_all(bind=engine)
    
    import_pre_drafted_questions()