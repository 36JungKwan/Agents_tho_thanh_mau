"""
Stress Test cho Thờ Mẫu RAG API.
Bắn TẤT CẢ requests ĐỒNG THỜI (không batch, không delay).
Hiển thị real-time khi từng request hoàn thành.

Cách dùng:
    python stress_test.py
"""
import asyncio
import aiohttp
import time
import statistics
from datetime import datetime

# ============================================================
# CẤU HÌNH TEST
# ============================================================
API_URL = "http://127.0.0.1:8000/api/chat"

TEST_QUESTIONS = [
    "Hầu giá Cô Bơ cần chuẩn bị gì?",
    "Thánh Mẫu Liễu Hạnh là ai?",
    "Ý nghĩa của nghi lễ Hầu Đồng là gì?",
    "Tứ Phủ trong Đạo Mẫu gồm những gì?",
    "Trang phục Hầu Đồng có ý nghĩa gì?",
    "Lễ Trình Đồng Mở Phủ diễn ra như thế nào?",
    "Vai trò của cung văn trong Hầu Đồng?",
    "Đạo Mẫu khác gì mê tín dị đoan?",
]

ARTISAN_ID = "087c9b25-4471-4323-9c5b-4092ba623810"
USER_ID = "0Dq1anfnUqVIRkdSO67ixdbUSE72"

# Tổng số request bắn ĐỒNG THỜI (tất cả cùng lúc, 0 delay)
TOTAL_REQUESTS = 20

# ============================================================
# MÀU SẮC TERMINAL
# ============================================================
class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
    CY = "\033[96m"; M = "\033[95m"; B = "\033[1m"; X = "\033[0m"

# ============================================================
# BIẾN TOÀN CỤC
# ============================================================
results_list = []
rate_limit_list = []
error_details = []
start_wall = 0
done_count = 0

# ============================================================
# GỬI 1 REQUEST + IN REAL-TIME
# ============================================================
async def fire_request(session, rid, question):
    global done_count

    payload = {
        "user_id": USER_ID,
        "session_id": f"stress_{rid}_{int(time.time())}",
        "artisan_id": ARTISAN_ID,
        "user_query": question
    }

    t0 = time.perf_counter()
    code = 0
    body = ""
    rate_limited = False
    err = ""

    try:
        async with session.post(API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            code = resp.status
            async for chunk in resp.content.iter_any():
                body += chunk.decode("utf-8", errors="ignore")

            if code == 429:
                rate_limited = True
            elif code == 500:
                low = body.lower()
                if any(k in low for k in ["rate limit", "429", "resource exhausted", "quota", "resourceexhausted"]):
                    rate_limited = True
    except asyncio.TimeoutError:
        err = "TIMEOUT (>120s)"
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:60]}"

    elapsed = time.perf_counter() - t0
    wall = time.perf_counter() - start_wall
    done_count += 1

    # Icon & status
    if err:
        icon, status = f"{C.R}💀", f"{C.R}{err}{C.X}"
    elif rate_limited:
        icon, status = f"{C.Y}⚠️", f"{C.Y}🚨 RATE LIMITED!{C.X}"
        rate_limit_list.append({"rid": rid, "wall": wall, "elapsed": elapsed})
    elif code == 200:
        icon, status = f"{C.G}✅", f"{C.G}OK ({len(body)} chars){C.X}"
    else:
        icon, status = f"{C.R}❌", f"{C.R}HTTP {code}{C.X}"
        # Lưu chi tiết lỗi 500
        error_details.append({"rid": rid, "code": code, "body": body[:200]})

    print(f"  {icon} [{done_count:>2}/{TOTAL_REQUESTS}] #{rid:<3} | ⏱️{elapsed:>5.1f}s | 🕐 T+{wall:>5.1f}s | {status}")

    results_list.append({
        "rid": rid, "code": code, "elapsed": elapsed, "wall": wall,
        "rate_limited": rate_limited, "chars": len(body), "err": err
    })

# ============================================================
# MAIN
# ============================================================
async def main():
    global start_wall, done_count
    done_count = 0

    print(f"\n{C.B}{'='*70}{C.X}")
    print(f"{C.B}{C.CY}  🔥 STRESS TEST - BẮN {TOTAL_REQUESTS} REQUESTS ĐỒNG THỜI{C.X}")
    print(f"{C.B}{'='*70}{C.X}")
    print(f"  URL:       {API_URL}")
    print(f"  Requests:  {C.B}{TOTAL_REQUESTS} (TẤT CẢ CÙNG LÚC, 0 delay){C.X}")
    print(f"  Artisan:   {ARTISAN_ID}")
    print(f"  User:      {USER_ID}")
    print(f"  Start:     {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    print(f"\n  {C.CY}⚡ BẮN!{C.X}\n")

    start_wall = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        # TẤT CẢ CÙNG LÚC - asyncio.gather bắn hết 1 phát
        tasks = [
            fire_request(session, i + 1, TEST_QUESTIONS[i % len(TEST_QUESTIONS)])
            for i in range(TOTAL_REQUESTS)
        ]
        await asyncio.gather(*tasks)

    total_wall = time.perf_counter() - start_wall

    # ============================================================
    # BÁO CÁO
    # ============================================================
    ok = [r for r in results_list if r["code"] == 200 and not r["rate_limited"]]
    rl = [r for r in results_list if r["rate_limited"]]
    fail = [r for r in results_list if r["err"] or (r["code"] not in [200, 429] and not r["rate_limited"])]

    print(f"\n{C.B}{'='*70}{C.X}")
    print(f"{C.B}{C.CY}  📊 BÁO CÁO TỔNG KẾT{C.X}")
    print(f"{C.B}{'='*70}{C.X}")
    print(f"\n  {C.G}✅ Thành công:    {len(ok)}/{TOTAL_REQUESTS}{C.X}")
    print(f"  {C.Y}⚠️  Rate Limited:  {len(rl)}/{TOTAL_REQUESTS}{C.X}")
    print(f"  {C.R}❌ Thất bại:      {len(fail)}/{TOTAL_REQUESTS}{C.X}")
    print(f"  🕐 Wall time:     {total_wall:.1f}s")

    if ok:
        times = [r["elapsed"] for r in ok]
        print(f"\n  ⏱️  Response time (thành công):")
        print(f"      Min:     {min(times):.1f}s")
        print(f"      Max:     {max(times):.1f}s")
        print(f"      Avg:     {statistics.mean(times):.1f}s")
        if len(times) > 1:
            print(f"      Median:  {statistics.median(times):.1f}s")
        print(f"\n  📈 Throughput: {len(ok) / total_wall * 60:.0f} req/phút")

    if rl:
        print(f"\n  {C.B}{C.Y}🚨 RATE LIMIT DETAILS:{C.X}")
        print(f"  {'─'*55}")
        for e in sorted(rl, key=lambda x: x["rid"]):
            print(f"    #{e['rid']:<3} | T+{e['wall']:>5.1f}s | response time: {e['elapsed']:.1f}s")
        print(f"  {'─'*55}")
        print(f"  {C.Y}→ Đầu tiên bị limit tại T+{min(e['wall'] for e in rl):.1f}s{C.X}")
    else:
        print(f"\n  {C.G}🎉 KHÔNG BỊ RATE LIMIT với {TOTAL_REQUESTS} requests đồng thời!{C.X}")

    # Chi tiết lỗi 500
    if error_details:
        print(f"\n  {C.B}{C.R}💀 CHI TIẾT LỖI 500:{C.X}")
        print(f"  {'─'*55}")
        for ed in error_details:
            print(f"    #{ed['rid']:<3} | HTTP {ed['code']}")
            print(f"    Body: {ed['body'][:150]}")
            print()
        print(f"  {'─'*55}")

    print(f"\n{C.B}{'='*70}{C.X}\n")

    # Ghi ra file để đọc lại
    with open("stress_test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"STRESS TEST REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total requests: {TOTAL_REQUESTS} (concurrent)\n")
        f.write(f"OK: {len(ok)} | Rate Limited: {len(rl)} | Failed: {len(fail)}\n")
        f.write(f"Wall time: {total_wall:.1f}s\n\n")
        if ok:
            times = [r["elapsed"] for r in ok]
            f.write(f"Response times: min={min(times):.1f}s max={max(times):.1f}s avg={statistics.mean(times):.1f}s\n")
            f.write(f"Throughput: {len(ok) / total_wall * 60:.0f} req/min\n\n")
        f.write("ALL RESULTS:\n")
        for r in sorted(results_list, key=lambda x: x["rid"]):
            status = "OK" if r["code"]==200 and not r["rate_limited"] else "RATE_LIMITED" if r["rate_limited"] else f"ERR_{r['code']}" if not r["err"] else r["err"]
            f.write(f"  #{r['rid']:<3} | {r['elapsed']:>5.1f}s | T+{r['wall']:>5.1f}s | {status} | {r['chars']} chars\n")
        if error_details:
            f.write(f"\nERROR 500 DETAILS:\n")
            for ed in error_details:
                f.write(f"  #{ed['rid']} | HTTP {ed['code']}\n  Body: {ed['body'][:300]}\n\n")
        if rl:
            f.write(f"\nRATE LIMIT DETAILS:\n")
            for e in sorted(rl, key=lambda x: x["rid"]):
                f.write(f"  #{e['rid']} | T+{e['wall']:.1f}s\n")
    print(f"  📄 Báo cáo đã lưu vào: stress_test_report.txt\n")

if __name__ == "__main__":
    print(f"\n{C.Y}⚡ Server phải đang chạy: uvicorn main:app --reload{C.X}")
    print(f"{C.Y}⚡ Ctrl+C để dừng{C.X}\n")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.R}🛑 Dừng bởi người dùng.{C.X}\n")
