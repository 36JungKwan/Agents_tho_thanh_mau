import os
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import func
from anthropic import Anthropic
from dotenv import load_dotenv

import models
from database import SessionLocal

load_dotenv()
claude_client = Anthropic(api_key=os.getenv("SECRET_API_KEY"))

def run_ai_b_coordinator():
    db: Session = SessionLocal()
    print("[AI Worker] Đang thức dậy và kiểm tra lịch làm việc...")
    
    try:
        artisans = db.query(models.Artisan).all()
        if not artisans:
            print("[AI Worker] Chưa có Nghệ nhân nào trong hệ thống. Tạm dừng.")
            return

        # ---------------------------------------------------------
        # LUỒNG 1: TÌM CÂU HỎI KHÓ TỪ NGƯỜI DÙNG
        # ---------------------------------------------------------
        unprocessed_questions = db.query(models.GlobalUnansweredQuestion).filter(
            models.GlobalUnansweredQuestion.is_processed_by_ai_b == False
        ).all()
        
        if unprocessed_questions:
            print(f"[AI Worker] Tìm thấy {len(unprocessed_questions)} câu hỏi thực tế. Đang chia bài...")
            for question in unprocessed_questions:
                # Viết lại câu hỏi lễ phép
                prompt = (
                    "Bạn là đệ tử đang học Đạo Mẫu. Một người dùng vừa hỏi: "
                    f"'{question.user_query}'.\n"
                    "Hãy chuyển tiếp câu này thành một câu hỏi cực kỳ lễ phép để hỏi Sư phụ (Nghệ nhân). "
                    "Xưng 'con', gọi 'Thầy/Cô'. Chỉ in ra câu hỏi."
                )
                response = claude_client.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                polite_prompt = response.content[0].text.strip()
                
                # Chia cho các Sư phụ
                for artisan in artisans:
                    db.add(models.InterviewQueue(
                        artisan_id=artisan.id, question_id=question.id,
                        ai_b_prompt=polite_prompt, status="pending"
                    ))
                question.is_processed_by_ai_b = True
                
            db.commit()
            print("[AI Worker] Đã chia xong câu hỏi từ Người dùng!")
            return # Dừng hàm vì đã hoàn thành KPI

        # ---------------------------------------------------------
        # LUỒNG 2: KHÔNG CÓ CÂU HỎI -> BỐC SÁCH RA HỎI (PROACTIVE)
        # ---------------------------------------------------------
        print("[AI Worker] Không có câu hỏi khó nào từ User. Đang trích xuất Sách để hỏi thăm Sư phụ...")
        
        # Bốc ngẫu nhiên (ORDER BY RANDOM) 1 đoạn chunk từ Sách phổ thông (owner_id = None)
        random_chunk = db.query(models.DocumentChunk).join(models.Document).filter(
            models.Document.owner_id == None
        ).order_by(func.random()).first()

        if random_chunk:
            print(f"Đã bốc được đoạn văn ở trang {random_chunk.page_number}: '{random_chunk.chunk_text[:50]}...'")
            
            # Yêu cầu Claude đọc đoạn sách và sinh ra câu hỏi gợi mở
            book_prompt = (
                "Bạn là đệ tử ngoan ngoãn đang học Đạo Mẫu. Bạn vừa đọc được đoạn sách sau:\n"
                f"--- SÁCH ---\n{random_chunk.chunk_text}\n---\n"
                "Nhiệm vụ: Hãy đặt MỘT câu hỏi lễ phép để hỏi Sư phụ (Nghệ nhân) về kinh nghiệm THỰC TẾ "
                "hoặc quan điểm riêng của Sư phụ về nội dung đoạn sách này (vì sách có thể viết chung chung). "
                "Xưng 'con', gọi 'Thầy/Cô'. Chỉ in ra câu hỏi."
            )
            
            response = claude_client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=200,
                messages=[{"role": "user", "content": book_prompt}]
            )
            polite_book_prompt = response.content[0].text.strip()
            print(f"Câu hỏi khơi gợi: '{polite_book_prompt}'")
            
            # Đẩy vào hộp thư Sư phụ (Lưu ý: question_id = None vì câu này không đến từ User)
            for artisan in artisans:
                db.add(models.InterviewQueue(
                    artisan_id=artisan.id, question_id=None,
                    ai_b_prompt=polite_book_prompt, status="pending"
                ))
            db.commit()
            print("[AI Worker] Đã giao bài tập khai thác kiến thức cho các Sư phụ!")

    except Exception as e:
        print(f"[AI Worker] Có lỗi: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    run_ai_b_coordinator()