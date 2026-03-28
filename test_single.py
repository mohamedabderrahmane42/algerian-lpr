"""
test_single.py  –  Visual test on photos from Matricules/
Draws YOLO bbox + CRNN plate text on each image and saves to output/test_results/
"""

import os, re, cv2, torch, numpy as np, torch.nn as nn

# ── reuse model classes from process_photos ──────────────────
PROJECT      = r"c:\Users\Mohamed\Desktop\projects\one"
INPUT_DIR    = os.path.join(PROJECT, "Matricules")
OUT_DIR      = os.path.join(PROJECT, "output", "test_results")
OCR_PATH     = os.path.join(PROJECT, "ocr_model.pt")
YOLO_PATHS   = [
    os.path.join(PROJECT, r"runs\detect\license_plate_det3\weights\best.pt"),
    os.path.join(PROJECT, r"runs\detect\license_plate_det2\weights\best.pt"),
    os.path.join(PROJECT, r"runs\detect\license_plate_det\weights\best.pt"),
]
CONF_THR     = 0.35
PADDING      = 12
N_SAMPLES    = 10          # number of photos to test

CHARSET  = '0123456789'
BLANK    = len(CHARSET)
NC       = len(CHARSET) + 1
IH, IW   = 32, 128

# ── CRNN ─────────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.rnn    = nn.LSTM(i, h, bidirectional=True, batch_first=False)
        self.linear = nn.Linear(h*2, o)
    def forward(self, x):
        o, _ = self.rnn(x)
        T,B,H = o.shape
        return self.linear(o.view(T*B,H)).view(T,B,-1)

class CRNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        def blk(ci,co,pk=(2,2),ps=(2,2)):
            return nn.Sequential(nn.Conv2d(ci,co,3,1,1),nn.BatchNorm2d(co),nn.ReLU(True),nn.MaxPool2d(pk,ps))
        self.cnn = nn.Sequential(blk(1,64),blk(64,128),blk(128,256,(2,1),(2,1)),
                                 blk(256,256,(2,1),(2,1)),blk(256,512,(2,1),(2,1)))
        self.rnn = nn.Sequential(BiLSTM(512,256,256), BiLSTM(256,256,nc))
    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(2,0,1)
        return torch.log_softmax(self.rnn(f), dim=2)

WILAYAS = {str(i).zfill(2) for i in range(1, 59)}

def decode(lp):
    ids = lp.argmax(1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK: out.append(int(i))
        prev = i
    return ''.join(CHARSET[i] for i in out if i < len(CHARSET))

def _try_fmt(d):
    """Return formatted plate if last 2 digits are a valid wilaya, else None."""
    if len(d) < 6: return None
    wil = d[-2:]
    if wil not in WILAYAS: return None
    mid = d[:-2]
    for rg in (3, 4):
        if len(mid) > rg:
            return f"{mid[:-rg]} {mid[-rg:]} {wil}"
    return f"{mid} {wil}"

def fmt(d):
    """
    Smart Algerian plate formatter with wilaya validation.
    Handles: 5+3+2 (new), 4+4+2 (older), and 11-digit OCR errors.
    """
    d = re.sub(r'[^0-9]', '', d)

    # Error-correction: prioritize 10-digit results if we have 11
    if len(d) == 11:
        for i in range(len(d)):
            result = _try_fmt(d[:i] + d[i+1:])
            if result:
                return result

    # Direct attempt
    result = _try_fmt(d)
    if result:
        return result

    # Hard fallback
    n = len(d)
    if n >= 10: return f"{d[:5]} {d[5:8]} {d[8:10]}"
    if n == 9:  return f"{d[:4]} {d[4:7]} {d[7:]}"
    return d

def preprocess(crop):
    ch,cw = crop.shape[:2]
    if cw < 100 or ch < 20:
        s = max(100/cw, 20/ch, 1.0)
        crop = cv2.resize(crop, (int(cw*s), int(ch*s)), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0,(4,4)).apply(g)
    g = cv2.resize(g,(IW,IH)).astype(np.float32)/255.
    return torch.FloatTensor(g).unsqueeze(0).unsqueeze(0)

# ── Draw helpers ─────────────────────────────────────────────
def draw_result(img, x1, y1, x2, y2, text, conf):
    # filled box border
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,200,0), 3)
    # label background
    label = f"{text}  ({conf:.2f})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    bg_y1 = max(y1-th-14, 0)
    cv2.rectangle(img, (x1, bg_y1), (x1+tw+8, y1), (0,200,0), -1)
    cv2.putText(img, label, (x1+4, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)
    return img

# ── Main ─────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load YOLO
    from ultralytics import YOLO
    yolo_path = next((p for p in YOLO_PATHS if os.path.exists(p)), None)
    if not yolo_path:
        print("ERROR: No YOLO model found – run train_model.py first.")
        return
    yolo = YOLO(yolo_path)
    print(f"YOLO loaded: {yolo_path}")

    # Load CRNN
    if not os.path.exists(OCR_PATH):
        print("ERROR: ocr_model.pt not found – run train_ocr.py first.")
        return
    ckpt  = torch.load(OCR_PATH, map_location=device)
    crnn  = CRNN(ckpt.get('num_classes', NC)).to(device)
    crnn.load_state_dict(ckpt['model_state_dict'])
    crnn.eval()
    print(f"CRNN loaded: {OCR_PATH}")

    # Pick N_SAMPLES images
    exts  = ('.jpg','.jpeg','.png')
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts)])
    # spread across the folder for variety
    step  = max(1, len(files)//N_SAMPLES)
    sample= files[::step][:N_SAMPLES]

    print(f"\nTesting on {len(sample)} images:\n")

    for fname in sample:
        img_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        results  = yolo(img, verbose=False)
        found    = False

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONF_THR: continue
                found = True

                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                x1p = max(0,x1-PADDING); y1p = max(0,y1-PADDING)
                x2p = min(W,x2+PADDING); y2p = min(H,y2+PADDING)
                crop = img[y1p:y2p, x1p:x2p]
                if crop.size==0: continue

                # CRNN
                t = preprocess(crop).to(device)
                with torch.no_grad():
                    lp = crnn(t)
                raw     = decode(lp[:,0,:].cpu().numpy())
                plate   = fmt(raw)

                # Annotate
                draw_result(img, x1, y1, x2, y2, plate, conf)
                print(f"  {fname:<35} ->  {plate}  (det={conf:.2f})")

        if not found:
            print(f"  {fname:<35}     [no plate detected]")

        # downscale for display if very large
        dh, dw = img.shape[:2]
        if dw > 1200:
            scale = 1200/dw
            img   = cv2.resize(img,(int(dw*scale),int(dh*scale)))

        out_path = os.path.join(OUT_DIR, f"result_{fname}")
        cv2.imwrite(out_path, img)

    print(f"\nSaved annotated images to: {OUT_DIR}")

if __name__ == '__main__':
    main()
