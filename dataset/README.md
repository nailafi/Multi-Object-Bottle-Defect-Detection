# 📁 CAE_dataset_autocrop

Dataset ini berisi contoh citra objek normal dalam satu *frame* yang diambil menggunakan webcam. Dataset tersebut terdiri atas tiga folder yang masing-masing berisi citra untuk komponen badan botol, tutup, dan label, serta digunakan sebagai data pelatihan pada setiap komponen.

# 📁 CAE_dataset_training

Kumpulan contoh citra pada folder **CAE_dataset_autocrop** kemudian dipotong (*cropping*) untuk menghilangkan latar belakang (*background*) sehingga hanya menyisakan area objek sesuai dengan komponen botol, yaitu badan botol, tutup, dan label. Hasil *cropping* tersebut selanjutnya digunakan sebagai dataset pelatihan model *Convolutional Autoencoder* (CAE).

# 📁 YOLO_labeling_roboflow

Kumpulan contoh data yang terdiri atas citra objek normal dan cacat untuk proses anotasi dengan memberikan *bounding box* pada setiap objek yang menjadi target deteksi. Hasil anotasi tersebut kemudian digunakan sebagai dataset pelatihan model YOLO.