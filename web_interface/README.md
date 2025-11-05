# Neural Network Steganography - Web Interface

🧠 **Hide Secret Text Messages in Neural Network Weights**

A user-friendly web application for embedding and extracting secret text messages in deep learning models using neural network steganography.

---

## 🚀 Quick Start

### 1. Start the Server

**Linux/Mac:**
```bash
./start_server.sh
```

**Windows:**
```
start_server.bat
```

**Or manually:**
```bash
python app.py
```

### 2. Open Browser

Navigate to: **http://localhost:5000**

### 3. Start Using!

- Click **"Embed Text"** to hide a message
- Click **"Extract Text"** to recover a message

---

## 📋 Features

### ✨ Text Embedding
- ✅ Embed secret messages (up to ~370 bytes)
- ✅ Support for various text formats: `.txt`, `.json`, `.py`, `.md`, `.java`, `.cpp`, `.js`, `.html`, `.css`, `.xml`, `.csv`
- ✅ Adjustable redundancy (10-50×)
- ✅ Adaptive redundancy for optimal survival
- ✅ Real-time quantization validation
- ✅ Download watermarked models

### ✨ Text Extraction
- ✅ Extract hidden messages from watermarked models
- ✅ Support for attack simulation (8-bit, 4-bit quantization)
- ✅ Confidence scoring for each bit
- ✅ Copy to clipboard or download as text file
- ✅ View extraction statistics

### ✨ Model Management
- ✅ Browse recent models
- ✅ Download model and metadata files
- ✅ View embedding statistics

---

## 💻 Usage Examples

### Example 1: Embed a Secret Message

1. Open browser: `http://localhost:5000`
2. Click **"Embed Text"**
3. Enter your message:
   ```
   This is my secret message hidden in a neural network!
   Serial: ABC-123-XYZ
   ```
4. Click **"Start Embedding"**
5. Wait ~5-10 minutes
6. Download both files:
   - `watermarked_model_YYYYMMDD_HHMMSS.pth`
   - `metadata_YYYYMMDD_HHMMSS.pkl`

### Example 2: Extract the Message

1. Click **"Extract Text"**
2. Upload both files (model + metadata)
3. Click **"Extract Message"**
4. See your recovered text in ~30 seconds!

### Example 3: Test Attack Resistance

1. Go to **"Extract Text"**
2. Upload files
3. Select **"4-bit Quantization"** attack
4. Click **"Extract Message"**
5. See survival statistics (should be ~87%)

---

## 📝 What Can You Embed?

### Text Messages
```
© 2025 YourName - Proprietary Model
License: Commercial Use Prohibited
Contact: your@email.com
```

### JSON Configuration
```json
{
  "model": "ResNet-18",
  "dataset": "CIFAR-10",
  "epochs": 100,
  "lr": 0.001
}
```

### Code Snippets
```python
def secret_function(x):
    return x ** 2 + 3 * x + 5
```

### Serial Numbers
```
MODEL-SERIAL: XYZ-789-2025-A1B2C3
```

### Anything text-based (up to ~370 characters)

---

## 📊 Expected Results

| Metric | Value | Description |
|--------|-------|-------------|
| **Embedding Time** | 5-10 min | With validation |
| **Max Text Size** | ~370 bytes | At 25× redundancy |
| **Accuracy Drop** | <1% | Model still works |
| **8-bit Survival** | 90-95% | After 8-bit quantization |
| **4-bit Survival** | 85-90% | After 4-bit quantization |
| **Clean Extraction** | 100% | Without attacks |

---

## 🔧 Configuration

### Change Port

Edit `app.py`:
```python
app.run(port=8080)  # Change from 5000 to 8080
```

### Change Parameters

Default embedding settings:
```python
redundancy = 25          # 10-50 recommended
step_multiplier = 0.25   # 0.10-0.50 recommended
adaptive = True          # ON for best results
validation = True        # ON to check quality
```

### File Locations

- **Models:** `models/`
- **Metadata:** `metadata/`
- **Uploads:** `uploads/` (temporary)

---

## 🆘 Troubleshooting

### ❌ Server won't start

**Solution:**
```bash
# Check if port is in use
lsof -i :5000

# Or use different port
python app.py  # Edit app.py to change port
```

### ❌ "CIFAR-10 download failed"

**Solution:**
- Check internet connection
- Dataset will download to `../data/` (~170MB)
- Only needed on first run

### ❌ "Model or metadata not found"

**Solution:**
- Upload **both** files (model.pth + metadata.pkl)
- Ensure files match (same embedding session)
- Check file extensions are correct

### ❌ Garbled extraction

**Solution:**
- Verify correct metadata file
- Try clean extraction (no attack)
- Re-download original files if corrupted

### ❌ Low survival rate

**Solution:**
- Increase redundancy to 30-40×
- Increase step multiplier to 0.30-0.40×
- Enable adaptive redundancy
- Use quantization validation

---

## 🔒 Security Notes

### ✅ What This Provides

- Statistical undetectability (passes chi-square test)
- Compression resistance (survives quantization/pruning)
- Model functionality preserved (accuracy drop <1%)

### ⚠️ What This Does NOT Provide

- Text encryption (message is not encrypted)
- Authentication (no signature verification)
- Active adversary protection (metadata reveals locations)

### 💡 Best Practices

1. **Encrypt sensitive data** before embedding
2. **Keep metadata secure** - required for extraction
3. **Use high redundancy** (30-40×) for critical data
4. **Test extraction** immediately after embedding
5. **Store backups** of both model and metadata files

---

## 📚 Documentation

- **This README:** Quick start guide
- **`../TECHNICAL_DOCUMENTATION.md`:** Complete technical details
- **`../WEB_INTERFACE_GUIDE.md`:** Comprehensive web interface guide

---

## 🎓 How It Works

### 1. Embedding Process

```
Your Text
    ↓
Convert to bytes (binary)
    ↓
Select robust weights in ResNet-18
    ↓
Apply QIM encoding (bit → weight parity)
    ↓
Repeat 25× for redundancy
    ↓
Watermarked Model + Metadata
```

### 2. Extraction Process

```
Watermarked Model + Metadata
    ↓
Read weight values at stored locations
    ↓
Apply QIM decoding (weight parity → bit)
    ↓
Majority voting (25 copies → 1 bit)
    ↓
Reconstruct bytes
    ↓
Recovered Text
```

### 3. Why It's Robust

- **25× Redundancy:** Each bit stored 25 times
- **Adaptive Allocation:** Vulnerable weights get extra protection
- **Soft-Decision Decoding:** Confidence-weighted voting
- **Robust Weight Selection:** High-importance weights don't get pruned

---

## 🎉 Success!

Your web interface is ready! Start embedding secret messages in neural networks at:

**http://localhost:5000**

For more details, see the complete documentation in `../TECHNICAL_DOCUMENTATION.md`

---

**Neural Network Steganography | 2025**
