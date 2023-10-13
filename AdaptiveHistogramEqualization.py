
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Bước 1: Load ảnh gốc
image_path = 'parrot.jpg'  # Thay thế bằng đường dẫn tới ảnh Parrot của bạn
image = cv2.imread(image_path)

# Bước 2: Chuyển ảnh sang thang độ xám
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bước 3: Áp dụng Cân bằng biểu đồ toàn cục
global_equalized = cv2.equalizeHist(gray_image)

# Bước 4: Tạo đối tượng CLAHE cho ô 8x8
clahe_8x8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_8x8_equalized = clahe_8x8.apply(gray_image)

# Bước 5: Tạo đối tượng CLAHE cho ô 16x16
clahe_16x16 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
clahe_16x16_equalized = clahe_16x16.apply(gray_image)

# Bước 6: Vẽ ảnh gốc và ảnh đã cân bằng
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Global Histogram Equalization")
plt.imshow(global_equalized, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Adaptive Histogram Equalization (8x8 tiles)")
plt.imshow(clahe_8x8_equalized, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Adaptive Histogram Equalization (16x16 tiles)")
plt.imshow(clahe_16x16_equalized, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

