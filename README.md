# Erosion & Dilation
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk membuat berbagai jenis struktur elemen
def get_structuring_elements():
    elements = {
        'Arbitrary': np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=np.uint8),
        'Octagon': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)),
        'Rectangle': cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)),
        'Diamond': np.array([[0, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 0],
                             [0, 0, 1, 0, 0]], dtype=np.uint8),
        'Line': cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)),
        'Pair': np.array([[1, 0],
                          [0, 1]], dtype=np.uint8),
        'Periodicline': np.array([[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]], dtype=np.uint8),
        'Disk': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)),
        'Square': cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    }
    return elements

# 1. Baca gambar dan konversi ke grayscale
img = cv2.imread('Prasasti2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Tingkatkan kontras dengan CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
contrast = clahe.apply(gray)

# 3. Thresholding adaptif (mean)
thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)

# 4. Terapkan Morfologi Closing dengan 9 bentuk kernel
elements = get_structuring_elements()
results = {}
for name, kernel in elements.items():
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Ubah tulisan menjadi putih, background menjadi hitam
    result = cv2.bitwise_not(closed)
    results[name] = result

# 5. Tampilkan hasil
plt.figure(figsize=(15, 12))
for i, (name, result) in enumerate(results.items()):
    plt.subplot(3, 3, i+1)
    plt.imshow(result, cmap='gray')
    plt.title(f'Strel: {name}')
    plt.axis('off')

plt.suptitle("Hasil Morfologi Closing dengan Berbagai Structuring Element\n(Tulisan Diputihkan, Background Dihilangkan)", fontsize=14)
plt.tight_layout()
plt.show()
```
