import fitz
import os
from PIL import Image

MAX_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB

assets_dir = os.path.dirname(os.path.abspath(__file__))

for fname in sorted(os.listdir(assets_dir)):
    if not fname.endswith('.pdf'):
        continue
    pdf_path = os.path.join(assets_dir, fname)
    png_path = os.path.join(assets_dir, fname.replace('.pdf', '.png'))

    doc = fitz.open(pdf_path)
    page = doc[0]

    # 从 150 DPI 开始，逐步降低直到文件 <= 2 MB
    dpi = 150
    while dpi >= 60:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(png_path)

        file_size = os.path.getsize(png_path)
        if file_size <= MAX_SIZE_BYTES:
            break

        # 文件仍超过 2MB，用 Pillow 进行 PNG 量化压缩
        img = Image.open(png_path).convert('P', palette=Image.ADAPTIVE, colors=256)
        img.save(png_path, optimize=True)
        file_size = os.path.getsize(png_path)
        if file_size <= MAX_SIZE_BYTES:
            break

        # 量化压缩后依然超限，降低 DPI 继续尝试
        dpi -= 30

    size_kb = os.path.getsize(png_path) / 1024
    print(f'✅ {fname}  ->  {os.path.basename(png_path)}  '
          f'({pix.width} x {pix.height} px, DPI={dpi}, {size_kb:.0f} KB)')
    doc.close()

print('\nAll PDFs converted.')
