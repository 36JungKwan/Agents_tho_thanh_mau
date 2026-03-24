import csv
import os
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

# Đảm bảo bảng artisans đã tồn tại
models.Base.metadata.create_all(bind=engine)

def import_artisans_from_csv(file_path: str):
    if not os.path.exists(file_path):
        print(f"Không tìm thấy file: {file_path}")
        return

    db: Session = SessionLocal()
    print(f"Đang đọc dữ liệu từ file '{file_path}'...")
    
    success_count = 0
    skip_count = 0

    try:
        with open(file_path, mode='r', encoding='utf-8-sig') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                name = row.get('name', '').strip()
                bio = row.get('bio', '').strip()
                
                # Bỏ qua các dòng trống trong file Excel
                if not name:
                    continue
                    
                # ---------------------------------------------------
                # BƯỚC CHECK TRÙNG LẶP (DUPLICATE CHECK)
                # ---------------------------------------------------
                existing_artisan = db.query(models.Artisan).filter(models.Artisan.name == name).first()
                
                if existing_artisan:
                    print(f"Bỏ qua: Nghệ nhân '{name}' đã tồn tại trong hệ thống (ID: {existing_artisan.id}).")
                    skip_count += 1
                    continue # Dừng xử lý dòng này, nhảy sang dòng tiếp theo
                # ---------------------------------------------------

                # Nếu chưa có thì mới tạo mới
                new_artisan = models.Artisan(
                    name=name,
                    bio=bio,
                    style_profile=None 
                )
                
                db.add(new_artisan)
                success_count += 1
                print(f"Đã xếp hàng nạp mới: {name}")

        # Đẩy toàn bộ dữ liệu lên Supabase
        db.commit()
        print(f"\nHOÀN TẤT QUÁ TRÌNH NẠP!")
        print(f"   - Thêm mới thành công: {success_count} người.")
        print(f"   - Bỏ qua (Đã tồn tại): {skip_count} người.")
        
    except Exception as e:
        db.rollback()
        print(f"\nLỗi trong quá trình nạp dữ liệu: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    csv_filename = "artisans_data.csv"
    import_artisans_from_csv(csv_filename)