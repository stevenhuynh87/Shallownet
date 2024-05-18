# import các thư viện cần thiết
# Cài đặt: pip install tensorflow==2.7.0
# Cài đặt pip install keras==2.7.0
# Nếu dùng phiên bản cao hơn thì lệnh 4, 5 phải thay đổi code
# Đây là file chính của chương trình để tạo ra model

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from conv.shallownet import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np


# Bước 1. Chuẩn bị dữ liệu
# Khởi tạo tiền xử lý ảnh
sp = SimplePreprocessor(32, 32) # Thiết lập kích thước ảnh 32 x 32
iap = ImageToArrayPreprocessor() # Gọi hàm để chuyển ảnh sang mảng

print("[INFO] Nạp ảnh...")
imagePaths = list(paths.list_images("datasets")) # tạo danh sách đường dẫn đến các folder con của folder datasets
# Nạp ảnh rồi chuyển mức xám của pixel trong vùng [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Chia tách dữ liệu vào 02 tập, training: 75% và testing: 25%
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)

# Chuyển dữ liệu nhãn ở số nguyên vào biểu diễn dưới dạng vectors
trainY = LabelBinarizer().fit_transform(trainY)  # Nhãn tập dữ liệu train
testY = LabelBinarizer().fit_transform(testY)    # Nhãn tâp dữ liệu test

# Bước 2. Xây dựng cấu trúc model (mạng)
# Tạo bộ tối ưu hóa cho model (hàm tối ưu SGD)
opt = SGD(learning_rate=0.005)

# Tạo model (mạng), biên dịch model
print("[INFO] Tạo mô hình...")
model = ShallowNet.build(width=32, height=32, depth=3, classes=3) # class = 3 là model phn 3 lớp
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# Bước 3. train model (mạng)
print("[INFO] training mạng ...")
#H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=32, epochs=10, verbose=1)
H = model.fit(trainX, trainY, validation_split = 0.1, batch_size=32, epochs=60, verbose=1)

# lưu model với tên: model.hdf5
model.save("model.hdf5")

# Bước 4. Đánh giá model (mạng)
print("[INFO] Đánh giá mạng...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=["cat", "dog", "panda"]))

# Vẽ kết quả train: Biểu đồ hàm loss quá trình train và độ chính xác (accuracy)
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 60), H.history["loss"], label="Model loss")
plt.plot(np.arange(0, 60), H.history["val_loss"], label="Validation loss")
plt.plot(np.arange(0, 60), H.history["accuracy"], label="Model accuracy")
plt.plot(np.arange(0, 60), H.history["val_accuracy"], label="Validation accuracy")
plt.title("Model loss and accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/accuracy")
plt.legend()
plt.show()