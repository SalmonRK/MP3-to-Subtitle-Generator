# Audio to SRT Generator (Thai Language)

แปลงไฟล์เสียง MP3 เป็นไฟล์คำบรรยาย SRT ภาษาไทย ด้วย AI 2 โมเดล:
- **Whisper Large** จาก OpenAI - รองรับหลายภาษา
- **Typhoon ASR** จาก SCB10X - เพิ่มความแม่นยำเฉพาะภาษาไทย

## การใช้งาน

### 1. การติดตั้งครั้งแรก
```bash
install.bat
```
จะติดตั้ง:
- สร้าง Virtual Environment
- ติดตั้ง Python packages ที่จำเป็น
- ดาวน์โหลดโมเดล Whisper (ใช้เวลาสักครู่)

### 2. การสร้างคำบรรยาย

#### วิธีง่ายที่สุด (ใช้ไฟล์ audio_file.mp3)
```bash
generate_srt.bat
```

#### สำหรับไฟล์เสียงอื่น
```bash
generate_srt_custom.bat "ชื่อไฟล์.mp3"
```

### 3. การใช้งานขั้นสูง (เลือกโมเดลเอง)

Activate environment ก่อน:
```bash
srt_env\Scripts\activate
```

ใช้ Whisper Large:
```bash
python audio_to_srt.py "ไฟล์เสียง.mp3" --model whisper --model-size large --verbose
```

ใช้ Typhoon ASR (แม่นยำสำหรับไทย):
```bash
python audio_to_srt.py "ไฟล์เสียง.mp3" --model typhoon --verbose
```

### 4. ทดสอบการใช้งาน Typhoon บน GPU

สำหรับทดสอบประสิทธิภาพของโมเดล Typhoon บน GPU:
```bash
python test_typhoon_simple.py
```

สคริปต์นี้จะ:
- ตรวจสอบข้อมูลระบบและ GPU
- ทดสอบโหลดโมเดล Typhoon บน GPU
- ทดสอบการแปลงเสียงเป็นข้อความภาษาไทย
- สร้างไฟล์ SRT พร้อมการแบ่งย่อหน้าที่เหมาะสม
- บันทึกรายงานผลการทดสอบ

Deactivate:
```bash
deactivate
```

## พารามิเตอร์ที่สำคัญ

| พารามิเตอร์ | คำอธิบาย | ตัวอย่าง |
|------------|----------|----------|
| `--model` | เลือกโมเดล (whisper/typhoon) | `--model whisper` |
| `--model-size` | ขนาดโมเดล Whisper | `--model-size large` |
| `--gpu` | ใช้ GPU (ถ้ามี) | `--gpu` |
| `--no-gpu` | ไม่ใช้ GPU | `--no-gpu` |
| `--verbose` | แสดงรายละเอียด | `--verbose` |

## ผลลัพธ์

- ไฟล์ SRT จะถูกสร้างในโฟลเดอร์ `output/`
- รองรับชื่อไฟล์ภาษาไทย
- ใช้ Encoding UTF-8 สำหรับภาษาไทย

## เปรียบเทียบโมเดล

| โมเดล | ความแม่นยำ (ภาษาไทย) | ความเร็ว | หน่วยความจำ |
|--------|---------------------|--------|------------|
| Whisper Large | ดี | เร็ว (GPU) | สูง (~3GB) |
| Typhoon ASR | ดีเยี่ยม | เร็วมาก (GPU) | ปานกลาง (~2GB) |

## ข้อควรรู้

- **Whisper**: เหมาะสำหรับเสียงหลายภาษา หรือมีคำศัพท์เฉพาะทาง
- **Typhoon**: เหมาะสำหรับเสียงไทยล้วน ต้องการความแม่นยำสูงสุด
- ต้องการ GPU เพื่อความเร็วในการประมวลผล
- ถ้าไม่มี GPU จะใช้ CPU แทน (ช้ากว่า)

## แก้ไขปัญหา

1. **ไม่เจอโมเดล**: รัน `install.bat` ใหม่
2. **ข้อความภาษาไทยผิดพลาด**: ลองเปลี่ยนโมเดล
3. **ช้าเกินไป**: ใช้ `--gpu` ถ้ามีการ์ดจอ
4. **ไฟล์เสียงไม่มีคำพูด**: ตรวจสอบคุณภาพเสียง

## โครงสร้างโฟลเดอร์

```
SRT-Generator/
├── models/           # โมเดล Whisper
├── srt_env/          # Virtual Environment
├── output/           # ไฟล์ SRT ที่สร้าง
├── audio_to_srt.py   # Script หลัก
├── install.bat       # ติดตั้งครั้งแรก
├── generate_srt.bat  # สร้างคำบรรยาย (ไฟล์ default)
└── generate_srt_custom.bat  # สร้างคำบรรยาย (ไฟล์อื่น)