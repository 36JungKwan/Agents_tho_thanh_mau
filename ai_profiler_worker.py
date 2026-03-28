import os
from sqlalchemy.orm import Session
from dotenv import load_dotenv

import models
from database import SessionLocal

load_dotenv()

# API Key Rotation Manager
from api_key_manager import key_manager

def run_ai_profiler():
    db: Session = SessionLocal()
    print("[AI PROFILER] Đang thức dậy để phân tích tâm lý Nghệ nhân...")

    try:
        artisans = db.query(models.Artisan).all()
        if not artisans:
            print("⚠️ [AI PROFILER] Chưa có Nghệ nhân nào trong hệ thống. Tạm dừng.")
            return

        for artisan in artisans:
            print(f"\n🔍 Đang phân tích hồ sơ Sư phụ: {artisan.name} (ID: {artisan.id})")
            
            # Kéo toàn bộ câu trả lời trong lịch sử của Sư phụ này
            all_answers = db.query(models.ArtisanAnswer).filter(
                models.ArtisanAnswer.artisan_id == artisan.id
            ).all()

            # Điều kiện: Cần ít nhất 1 câu trả lời mới đủ dữ liệu để phân tích tính cách
            if not all_answers or len(all_answers) < 1:
                print(f"  -> Bỏ qua: Sư phụ chưa có đủ dữ liệu (hiện có {len(all_answers)} câu).")
                continue

            # Gộp tất cả di sản Text lại thành một khối
            combined_text = "\n".join([f"- {ans.answer_text}" for ans in all_answers])
            
            # Khởi tạo Prompt cho Chuyên gia tâm lý học (Gemini)
            profiler_prompt = (
                "Bạn là một chuyên gia phân tích ngôn ngữ học và tâm lý học hành vi. "
                "Hãy đọc các câu nói thực tế sau của một Nghệ nhân Đạo Mẫu:\n"
                f"{combined_text}\n\n"
                "Nhiệm vụ: Hãy viết một đoạn MIÊU TẢ ĐẦY ĐỦ (dưới 300 chữ) về phong cách giao tiếp, khẩu khí, "
                "cách xưng hô thói quen, và nhịp điệu câu từ của người này. "
                "Đoạn miêu tả này sẽ được dùng trực tiếp làm System Prompt để một AI khác nhập vai (Roleplay) chính người này. "
                "TUYỆT ĐỐI CHỈ IN RA ĐOẠN MIÊU TẢ, KHÔNG GỞI LỜI CHÀO HAY GIẢI THÍCH GÌ THÊM."
            )

            # Gọi Gemini 2.5 Flash thực thi
            response = key_manager.generate_with_retry(
                model='gemini-2.5-flash',
                contents=profiler_prompt
            )

            profile_result = response.text.strip() 
            
            # Lưu/Cập nhật đoạn "Khí chất" này vào Database
            artisan.style_profile = profile_result
            print(f"Đã đúc kết xong khí chất:\n  '{profile_result}'")
            
        # Commit toàn bộ thay đổi lên DB sau khi quét xong tất cả Nghệ nhân
        db.commit()
        print("\n[AI PROFILER] Đã hoàn tất việc cập nhật Khí chất cho toàn bộ Nghệ nhân!")

    except Exception as e:
        print(f"[AI PROFILER] Có lỗi xảy ra: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    run_ai_profiler()