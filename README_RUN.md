# 🚀 Hướng Dẫn Chạy Dự Án RAG-Anything (Phiên Bản Tối Ưu Cho Windows & CPU)

Tài liệu này hướng dẫn bạn cách khởi chạy và sử dụng `RAG-Anything` một cách mượt mà nhất trên môi trường Windows (không cần GPU), sử dụng parser `simple_docx` siêu nhẹ được trợ lực bởi **Docling API**.

---

## 1. Yêu Cầu Hệ Thống & Môi Trường

Bạn cần đảm bảo file `.env` đã được cấu hình đúng để sử dụng **Gemini Flash-Lite** (tiết kiệm Quota) và **Ollama Local** (cho Embedding).

Mở file `.env` và đảm bảo các biến sau được thiết lập:

```ini
# LLM Provider (Sử dụng Gemini)
LLM_MODEL=gemini-3.1-flash-lite
VISION_MODEL=gemini-3.1-flash-lite
FALLBACK_VISION_MODEL=gemini-2.5-flash-lite
GEMINI_API_KEY=your_gemini_api_key_here

# Embedding Provider (Sử dụng Ollama cục bộ để miễn phí & không giới hạn Quota)
EMBEDDING_PROVIDER=ollama
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=nomic-embed-local:latest
OLLAMA_HOST=http://localhost:11434
EMBEDDING_DIM=768
```

*Lưu ý: Bạn phải cài đặt và chạy ứng dụng Ollama trên máy tính với model `nomic-embed-local:latest`.*

---

## 2. Cài Đặt Thư Viện Cần Thiết

Vì môi trường Windows thuần CPU thường hay gặp lỗi khi xử lý các model nặng, dự án đã được thiết kế lại để cài đặt các phiên bản siêu nhẹ.

Mở PowerShell tại thư mục dự án (`D:\TMA\RAG-Anything`) và chạy lần lượt các lệnh sau:

```powershell
# Bật môi trường ảo (Virtual Environment)
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install docling
```

---

## 3. Khởi Chạy Nạp Dữ Liệu (Ingestion)

Để đưa một tài liệu `.docx` vào hệ thống RAG (ví dụ: `LuatLaoDong2012.docx`), bạn sử dụng công cụ `raganything_example.py` với cờ `--parser simple_docx`.

```powershell
python .\examples\raganything_example.py .\inputs\LuatLaoDong2012.docx --parser simple_docx --working_dir .\rag_storage\luatld --output .\output\luatld --skip-query
```
python .\examples\raganything_example.py .\inputs\LuatLaoDong2012.docx --parser docling --working_dir .\rag_storage\luatld --output .\output\luatld --skip-query


**Giải thích các tham số:**
- `.\inputs\LuatLaoDong2012.docx`: File đầu vào của bạn.
- `--parser simple_docx`: Yêu cầu hệ thống sử dụng Docling nhẹ (đã cấu hình bỏ qua xử lý ảnh/OCR để tiết kiệm CPU & RAM).
- `--working_dir .\rag_storage\luatld`: Thư mục lưu trữ Vector DB và Graph DB.
- `--output .\output\luatld`: Thư mục lưu file Markdown đã được trích xuất để bạn có thể xem lại cấu trúc.
- `--skip-query`: Chỉ thực hiện bước đọc hiểu (Ingestion), chưa truy vấn ngay lập tức.

*Lưu ý: Hệ thống đã tự động trỏ thư mục Cache (HF_HOME) về `.tmp/hf` trong dự án để tránh hoàn toàn lỗi phân quyền `WinError 1314` trên Windows.*

---

## 4. Xử Lý Đa Phương Thức (PDF, Hình Ảnh, PPTX...)

Nếu bạn muốn xử lý các loại file phức tạp hơn có chứa **Hình ảnh, Bảng biểu, Công thức Toán học**, hoặc các định dạng file khác như `.pdf`, `.pptx`, `.jpg` (không chỉ riêng `.docx`), hãy thay đổi cờ thành `--parser docling`.

Hệ thống sẽ tự động bóc tách hình ảnh và gửi cho **Gemini Vision (Flash-Lite)** để phân tích, trong khi **Ollama** vẫn đóng vai trò nhúng văn bản. Bạn không cần tải thêm bất kỳ model Vision nặng nề nào cả!

```powershell
python .\examples\raganything_example.py .\inputs\Ten_File_Cua_Ban.pdf --parser docling --working_dir .\rag_storage\my_data --skip-query
```

*Lưu ý khi chạy đa phương thức:*
- **RAM tiêu thụ sẽ cao hơn:** Vì Docling sẽ phải chạy OCR và trích xuất hình ảnh.
- **Tải model tự động:** Trong lần chạy ĐẦU TIÊN, Docling sẽ tự động tải một số model bố cục (layout analysis) siêu nhỏ gọn từ HuggingFace về thư mục `.tmp/hf`. Việc này hoàn toàn tự động, bạn chỉ cần kiên nhẫn chờ (tốn khoảng 1-3 phút tuỳ mạng).

---

## 5. Truy Vấn (Querying)

Khi dữ liệu đã được nạp thành công, bạn chỉ cần chạy lại lệnh và **bỏ cờ `--skip-query`** đi.

```powershell
python .\examples\raganything_example.py .\inputs\LuatLaoDong2012.docx --parser simple_docx --working_dir .\rag_storage\luatld
```

Lúc này, script sẽ tự động chạy qua các câu hỏi mẫu (hardcoded) có sẵn trong file `raganything_example.py` (ví dụ: *"Trong Bộ luật Lao động 2012, Điều 1 quy định về nội dung gì?"*) và trả về kết quả truy vấn RAG.

*(Mẹo: Bạn có thể mở file `examples/raganything_example.py` ở dòng 553 để sửa lại danh sách các câu hỏi `text_queries` theo ý muốn của riêng bạn!)*

## 6. Xử Lý Sự Cố Thường Gặp (Troubleshooting)

- **Lỗi `429 RESOURCE_EXHAUSTED` (Gemini):** Bạn đã dùng hết 20 request/ngày của tài khoản Free. Cách giải quyết là chờ qua ngày, dùng tài khoản trả phí, hoặc chia nhỏ file `.docx` ra.
- **Lỗi `Docling check_installation failed`:** Thường do bạn chưa cài đủ lệnh PyTorch ở bước 2. Hãy chạy lại lệnh cài đặt PyTorch `--index-url https://download.pytorch.org/whl/cpu`.
- **Hệ thống bị treo khi gọi Ollama:** Hãy chắc chắn biểu tượng Ollama đang chạy dưới khay hệ thống (System Tray) và bạn đã từng chạy `ollama pull nomic-embed-local:latest` trong terminal.

## 9. Chạy React/Vite Frontend Đúng Thư Mục

Frontend React/Vite nằm trong:

```text
frontend/app
```

Vì vậy phải chạy `npm` trong đúng thư mục này:

```powershell
cd D:\TMA\RAG-Anything\frontend\app
npm install
npm run dev
```

Không chạy `npm install` hoặc `npm run dev` ở thư mục gốc `D:\TMA\RAG-Anything`, vì thư mục gốc không có `package.json`.

---

## 7. Chạy Trực Tiếp Từ Terminal Với `--query` (PowerShell)

Bạn có thể truyền câu hỏi trực tiếp từ terminal mà không cần sửa file Python: script đã hỗ trợ tuỳ chọn `--query` (viết tắt `-q`) — có thể lặp lại để gửi nhiều câu hỏi.

Ví dụ (PowerShell):

```powershell
.\.venv\Scripts\python.exe .\examples\raganything_example.py .\inputs\dog.jpg --parser paddleocr --working_dir .\rag_storage\dog --query "What is this picture about?"
```
```powershell
.\.venv\Scripts\python.exe .\examples\raganything_example.py .\inputs\2509_taai_cognitive_load_final.pdf --parser pdf_hybrid --working_dir .\rag_storage\cognitive --query "What is Fig 2 about?"
```
```powershell
.\.venv\Scripts\python.exe .\examples\raganything_example.py .\inputs\2509_taai_cognitive_load_final.pdf --parser pdf_hybrid --vision-target "Fig. 2" --max-vision-pages 1 --working_dir .\rag_storage\cognitive_fig2 --query "What is Fig 2 about?"
```
```powershell
.\.venv\Scripts\python.exe .\examples\raganything_example.py .\inputs\2509_taai_cognitive_load_final.pdf --parser pdf_hybrid --page-range 11 --no-vision --skip-kg-extraction --working_dir .\rag_storage\cognitive_table2 --query "In Table 2, what are the Accuracy and ROC values for MLP at Scale 2? Answer only from the table."
```
```powershell
.\.venv\Scripts\python.exe .\examples\raganything_example.py .\inputs\2509_taai_cognitive_load_final.pdf --parser pdf_hybrid --no-vision --skip-kg-extraction --working_dir .\rag_storage\cognitive_equation --query "What is the Accuracy formula shown in the paper? Answer only with the equation."
```

Bạn cũng có thể dùng luôn dạng đầy đủ `--query` (không cần viết tắt `-q`) trong lệnh, ví dụ chính xác như sau:

```powershell
PS D:\TMA\RAG-Anything> python .\examples\raganything_example.py .\inputs\pedestrian.png --parser paddleocr --working_dir .\rag_storage\pedestrian --query "Trong bức ảnh này có bao nhiêu người đi bộ, và họ đang làm gì?"
```

Lưu ý ngắn:
- `--query` có thể lặp lại nếu muốn gửi nhiều câu hỏi.
- Trên PowerShell dùng `"..."` cho chuỗi có dấu cách. Nếu câu hỏi chứa dấu ngoặc kép, hãy dùng single quotes `'...'` hoặc escape ký tự " bằng `\"`.

---
## 8. Chạy WebUI Local (Gradio)

Nếu bạn muốn dùng giao diện web thay vì gõ lệnh CLI, chạy:

```powershell
.\.venv\Scripts\python.exe .\examples\webui_gradio.py
```

Sau đó mở trình duyệt tại:

```text
http://127.0.0.1:7860
```

Luồng sử dụng cơ bản:
- Bước 1: Upload file (`.pdf`, `.docx`, `.png/.jpg`, ...).
- Bước 2: Bấm **Process / Index** để nạp dữ liệu 1 lần vào `rag_storage`.
- Bước 3: Dùng khung chat để hỏi nhiều câu liên tiếp.

Lưu ý quan trọng:
- Sau khi đã index, các câu hỏi tiếp theo chỉ query trên index hiện có, không parse/index lại mỗi lần hỏi.
- Với PDF, mặc định WebUI ưu tiên tiết kiệm quota (`no_vision=True`, `skip_kg_extraction=True`, `max_vision_pages=1`).
- Nếu cần phân tích hình cụ thể trong PDF, dùng phần **Analyze visual target** (ví dụ `Fig. 2`) hoặc `vision_page_range`.

---
# Xem log tất cả services
docker compose logs

# Theo dõi log realtime
docker compose logs -f

# Chỉ xem log rag-webui
docker compose logs rag-webui

# Chỉ xem 200 dòng gần nhất của rag-webui
docker compose logs --tail=200 rag-webui

# Realtime + 200 dòng gần nhất
docker compose logs -f --tail=200 rag-webui
---
*Chúc bạn trải nghiệm RAG-Anything phiên bản siêu việt một cách mượt mà nhất!*
