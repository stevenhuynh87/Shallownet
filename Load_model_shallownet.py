# import thư viện cần thiết
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2

# Khởi tạo danh sách nhãn
classLabels = ["cat", "dog", "panda"]

print("[INFO] Đang nạp ảnh mẫu để phân lớp...")
imagePaths = np.array(list(paths.list_images("image"))) # tạo danh sách đường dẫn đến file ảnh trong folder image
idxs = np.random.randint(0, len(imagePaths), size=(10,)) # Trả về 5 số nguyên ngẫu nhiên tương ứng với đường dẫn đến file ảnh
imagePaths = imagePaths[idxs] # Tạo danh sách chứa 10 số nguyên ngẫu nhiên

# Khởi tạo tiền xử lý các file ảnh
sp = SimplePreprocessor(32, 32) # Thiết lập kích thước ảnh 32 x 32
iap = ImageToArrayPreprocessor() # Gọi hàm để chuyển ảnh sang mảng

# Nạp ảnh rồi chuyển mức xám của pixel trong vùng [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# load model đã được train
print("[INFO] Nạp model mạng pre-trained ...")
model = load_model("model.hdf5")

# Dự đoán
print("[INFO] Đang dự đoán để phân lớp...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# Lặp qua tất cả các file ảnh trong imagePaths và thực hiện trên từng ảnh, gồm:
# Nạp ảnh --> tạo label dự đoán trên ảnh --> Hiển thị ảnh
for (i, imagePath) in enumerate(imagePaths):
    # Đọc file ảnh
    image = cv2.imread(imagePath)
    # Vẽ label dự đoán lên ảnh
    cv2.putText(image, "label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Hiển thị ảnh
    cv2.imshow("Image", image)
    cv2.waitKey(0)







