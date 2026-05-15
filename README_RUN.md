# ðŸš€ HÆ°á»›ng Dáº«n Cháº¡y Dá»± Ãn RAG-Anything (PhiÃªn Báº£n Tá»‘i Æ¯u Cho Windows & CPU)

TÃ i liá»‡u nÃ y hÆ°á»›ng dáº«n báº¡n cÃ¡ch khá»Ÿi cháº¡y vÃ  sá»­ dá»¥ng `RAG-Anything` má»™t cÃ¡ch mÆ°á»£t mÃ  nháº¥t trÃªn mÃ´i trÆ°á»ng Windows (khÃ´ng cáº§n GPU), sá»­ dá»¥ng parser `simple_docx` siÃªu nháº¹ Ä‘Æ°á»£c trá»£ lá»±c bá»Ÿi **Docling API**.

---

## 1. YÃªu Cáº§u Há»‡ Thá»‘ng & MÃ´i TrÆ°á»ng

Báº¡n cáº§n Ä‘áº£m báº£o file `.env` Ä‘Ã£ Ä‘Æ°á»£c cáº¥u hÃ¬nh Ä‘Ãºng Ä‘á»ƒ sá»­ dá»¥ng **Gemini Flash-Lite** (tiáº¿t kiá»‡m Quota) vÃ  **Ollama Local** (cho Embedding).

Má»Ÿ file `.env` vÃ  Ä‘áº£m báº£o cÃ¡c biáº¿n sau Ä‘Æ°á»£c thiáº¿t láº­p:

```ini
# LLM Provider (Sá»­ dá»¥ng Gemini)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash-lite
GEMINI_API_KEY=your_gemini_api_key_here

# Embedding Provider (Sá»­ dá»¥ng Ollama cá»¥c bá»™ Ä‘á»ƒ miá»…n phÃ­ & khÃ´ng giá»›i háº¡n Quota)
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-local
EMBEDDING_DIM=768
```

*LÆ°u Ã½: Báº¡n pháº£i cÃ i Ä‘áº·t vÃ  cháº¡y á»©ng dá»¥ng Ollama trÃªn mÃ¡y tÃ­nh vá»›i model `nomic-embed-text`.*

---

## 2. CÃ i Äáº·t ThÆ° Viá»‡n Cáº§n Thiáº¿t

VÃ¬ mÃ´i trÆ°á»ng Windows thuáº§n CPU thÆ°á»ng hay gáº·p lá»—i khi xá»­ lÃ½ cÃ¡c model náº·ng, dá»± Ã¡n Ä‘Ã£ Ä‘Æ°á»£c thiáº¿t káº¿ láº¡i Ä‘á»ƒ cÃ i Ä‘áº·t cÃ¡c phiÃªn báº£n siÃªu nháº¹.

Má»Ÿ PowerShell táº¡i thÆ° má»¥c dá»± Ã¡n (`D:\TMA\RAG-Anything`) vÃ  cháº¡y láº§n lÆ°á»£t cÃ¡c lá»‡nh sau:

```powershell
# Báº­t mÃ´i trÆ°á»ng áº£o (Virtual Environment)
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install docling
```

---

## 3. Khá»Ÿi Cháº¡y Náº¡p Dá»¯ Liá»‡u (Ingestion)

Äá»ƒ Ä‘Æ°a má»™t tÃ i liá»‡u `.docx` vÃ o há»‡ thá»‘ng RAG (vÃ­ dá»¥: `LuatLaoDong2012.docx`), báº¡n sá»­ dá»¥ng cÃ´ng cá»¥ `raganything_example.py` vá»›i cá» `--parser simple_docx`.

```powershell
python .\examples\raganything_example.py .\inputs\LuatLaoDong2012.docx --parser simple_docx --working_dir .\rag_storage\luatld --output .\output\luatld --skip-query
```
python .\examples\raganything_example.py .\inputs\LuatLaoDong2012.docx --parser docling --working_dir .\rag_storage\luatld --output .\output\luatld --skip-query


**Giáº£i thÃ­ch cÃ¡c tham sá»‘:**
- `.\inputs\LuatLaoDong2012.docx`: File Ä‘áº§u vÃ o cá»§a báº¡n.
- `--parser simple_docx`: YÃªu cáº§u há»‡ thá»‘ng sá»­ dá»¥ng Docling nháº¹ (Ä‘Ã£ cáº¥u hÃ¬nh bá» qua xá»­ lÃ½ áº£nh/OCR Ä‘á»ƒ tiáº¿t kiá»‡m CPU & RAM).
- `--working_dir .\rag_storage\luatld`: ThÆ° má»¥c lÆ°u trá»¯ Vector DB vÃ  Graph DB.
- `--output .\output\luatld`: ThÆ° má»¥c lÆ°u file Markdown Ä‘Ã£ Ä‘Æ°á»£c trÃ­ch xuáº¥t Ä‘á»ƒ báº¡n cÃ³ thá»ƒ xem láº¡i cáº¥u trÃºc.
- `--skip-query`: Chá»‰ thá»±c hiá»‡n bÆ°á»›c Ä‘á»c hiá»ƒu (Ingestion), chÆ°a truy váº¥n ngay láº­p tá»©c.

*LÆ°u Ã½: Há»‡ thá»‘ng Ä‘Ã£ tá»± Ä‘á»™ng trá» thÆ° má»¥c Cache (HF_HOME) vá» `.tmp/hf` trong dá»± Ã¡n Ä‘á»ƒ trÃ¡nh hoÃ n toÃ n lá»—i phÃ¢n quyá»n `WinError 1314` trÃªn Windows.*

---

## 4. Xá»­ LÃ½ Äa PhÆ°Æ¡ng Thá»©c (PDF, HÃ¬nh áº¢nh, PPTX...)

Náº¿u báº¡n muá»‘n xá»­ lÃ½ cÃ¡c loáº¡i file phá»©c táº¡p hÆ¡n cÃ³ chá»©a **HÃ¬nh áº£nh, Báº£ng biá»ƒu, CÃ´ng thá»©c ToÃ¡n há»c**, hoáº·c cÃ¡c Ä‘á»‹nh dáº¡ng file khÃ¡c nhÆ° `.pdf`, `.pptx`, `.jpg` (khÃ´ng chá»‰ riÃªng `.docx`), hÃ£y thay Ä‘á»•i cá» thÃ nh `--parser docling`.

Há»‡ thá»‘ng sáº½ tá»± Ä‘á»™ng bÃ³c tÃ¡ch hÃ¬nh áº£nh vÃ  gá»­i cho **Gemini Vision (Flash-Lite)** Ä‘á»ƒ phÃ¢n tÃ­ch, trong khi **Ollama** váº«n Ä‘Ã³ng vai trÃ² nhÃºng vÄƒn báº£n. Báº¡n khÃ´ng cáº§n táº£i thÃªm báº¥t ká»³ model Vision náº·ng ná» nÃ o cáº£!

```powershell
python .\examples\raganything_example.py .\inputs\Ten_File_Cua_Ban.pdf --parser docling --working_dir .\rag_storage\my_data --skip-query
```

*LÆ°u Ã½ khi cháº¡y Ä‘a phÆ°Æ¡ng thá»©c:*
- **RAM tiÃªu thá»¥ sáº½ cao hÆ¡n:** VÃ¬ Docling sáº½ pháº£i cháº¡y OCR vÃ  trÃ­ch xuáº¥t hÃ¬nh áº£nh.
- **Táº£i model tá»± Ä‘á»™ng:** Trong láº§n cháº¡y Äáº¦U TIÃŠN, Docling sáº½ tá»± Ä‘á»™ng táº£i má»™t sá»‘ model bá»‘ cá»¥c (layout analysis) siÃªu nhá» gá»n tá»« HuggingFace vá» thÆ° má»¥c `.tmp/hf`. Viá»‡c nÃ y hoÃ n toÃ n tá»± Ä‘á»™ng, báº¡n chá»‰ cáº§n kiÃªn nháº«n chá» (tá»‘n khoáº£ng 1-3 phÃºt tuá»³ máº¡ng).

---

## 5. Truy Váº¥n (Querying)

Khi dá»¯ liá»‡u Ä‘Ã£ Ä‘Æ°á»£c náº¡p thÃ nh cÃ´ng, báº¡n chá»‰ cáº§n cháº¡y láº¡i lá»‡nh vÃ  **bá» cá» `--skip-query`** Ä‘i.

```powershell
python .\examples\raganything_example.py .\inputs\LuatLaoDong2012.docx --parser simple_docx --working_dir .\rag_storage\luatld
```

LÃºc nÃ y, script sáº½ tá»± Ä‘á»™ng cháº¡y qua cÃ¡c cÃ¢u há»i máº«u (hardcoded) cÃ³ sáºµn trong file `raganything_example.py` (vÃ­ dá»¥: *"Trong Bá»™ luáº­t Lao Ä‘á»™ng 2012, Äiá»u 1 quy Ä‘á»‹nh vá» ná»™i dung gÃ¬?"*) vÃ  tráº£ vá» káº¿t quáº£ truy váº¥n RAG.

*(Máº¹o: Báº¡n cÃ³ thá»ƒ má»Ÿ file `examples/raganything_example.py` á»Ÿ dÃ²ng 553 Ä‘á»ƒ sá»­a láº¡i danh sÃ¡ch cÃ¡c cÃ¢u há»i `text_queries` theo Ã½ muá»‘n cá»§a riÃªng báº¡n!)*

## 6. Xá»­ LÃ½ Sá»± Cá»‘ ThÆ°á»ng Gáº·p (Troubleshooting)

- **Lá»—i `429 RESOURCE_EXHAUSTED` (Gemini):** Báº¡n Ä‘Ã£ dÃ¹ng háº¿t 20 request/ngÃ y cá»§a tÃ i khoáº£n Free. CÃ¡ch giáº£i quyáº¿t lÃ  chá» qua ngÃ y, dÃ¹ng tÃ i khoáº£n tráº£ phÃ­, hoáº·c chia nhá» file `.docx` ra.
- **Lá»—i `Docling check_installation failed`:** ThÆ°á»ng do báº¡n chÆ°a cÃ i Ä‘á»§ lá»‡nh PyTorch á»Ÿ bÆ°á»›c 2. HÃ£y cháº¡y láº¡i lá»‡nh cÃ i Ä‘áº·t PyTorch `--index-url https://download.pytorch.org/whl/cpu`.
- **Há»‡ thá»‘ng bá»‹ treo khi gá»i Ollama:** HÃ£y cháº¯c cháº¯n biá»ƒu tÆ°á»£ng Ollama Ä‘ang cháº¡y dÆ°á»›i khay há»‡ thá»‘ng (System Tray) vÃ  báº¡n Ä‘Ã£ tá»«ng cháº¡y `ollama pull nomic-embed-text` trong terminal.

---

## 7. Cháº¡y Trá»±c Tiáº¿p Tá»« Terminal Vá»›i `--query` (PowerShell)

Báº¡n cÃ³ thá»ƒ truyá»n cÃ¢u há»i trá»±c tiáº¿p tá»« terminal mÃ  khÃ´ng cáº§n sá»­a file Python: script Ä‘Ã£ há»— trá»£ tuá»³ chá»n `--query` (viáº¿t táº¯t `-q`) â€” cÃ³ thá»ƒ láº·p láº¡i Ä‘á»ƒ gá»­i nhiá»u cÃ¢u há»i.

VÃ­ dá»¥ (PowerShell):

```powershell
.\.venv\Scripts\python.exe .\examples\raganything_example.py .\inputs\dog.jpg --parser paddleocr --working_dir .\rag_storage\dog --query "Trong áº£nh cÃ³ gÃ¬?"
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

Báº¡n cÅ©ng cÃ³ thá»ƒ dÃ¹ng luÃ´n dáº¡ng Ä‘áº§y Ä‘á»§ `--query` (khÃ´ng cáº§n viáº¿t táº¯t `-q`) trong lá»‡nh, vÃ­ dá»¥ chÃ­nh xÃ¡c nhÆ° sau:

```powershell
PS D:\TMA\RAG-Anything> python .\examples\raganything_example.py .\inputs\pedestrian.png --parser paddleocr --working_dir .\rag_storage\pedestrian --query "Trong bá»©c áº£nh nÃ y cÃ³ bao nhiÃªu ngÆ°á»i Ä‘i bá»™, vÃ  há» Ä‘ang lÃ m gÃ¬?"
```

LÆ°u Ã½ ngáº¯n:
- `--query` cÃ³ thá»ƒ láº·p láº¡i náº¿u muá»‘n gá»­i nhiá»u cÃ¢u há»i.
- TrÃªn PowerShell dÃ¹ng `"..."` cho chuá»—i cÃ³ dáº¥u cÃ¡ch. Náº¿u cÃ¢u há»i chá»©a dáº¥u ngoáº·c kÃ©p, hÃ£y dÃ¹ng single quotes `'...'` hoáº·c escape kÃ½ tá»± " báº±ng `\"`.

---
*ChÃºc báº¡n tráº£i nghiá»‡m RAG-Anything phiÃªn báº£n siÃªu viá»‡t má»™t cÃ¡ch mÆ°á»£t mÃ  nháº¥t!*
