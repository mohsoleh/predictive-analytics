# LAPORAN PROYEK MACHINE LEARNING - Moh Saleh
## Domain Proyek
Pendidikan memiliki peran krusial dalam kehidupan suatu negara untuk memastikan kelangsungan hidup bangsa. Menurut penelitian yang dilakukan oleh Paulo Cortez dan Silvia, tingkat pendidikan di Portugal telah mengalami peningkatan selama sepuluh tahun terakhir. Namun, statistik menunjukkan bahwa Portugal masih berada di peringkat bawah dalam pendidikan, yang disebabkan oleh tingginya angka putus sekolah. Faktor utama siswa putus sekolah di Portugal berkaitan dengan kegagalan mereka dalam menyelesaikan bidang studi tertentu, terutama matematika dan bahasa Portugis ([Cortez & Silva, 2008](https://repositorium.sdum.uminho.pt/handle/1822/8024)). 

Ada banyak  cara  yang  dapat  digunakan untuk  menganalisis  kemampuan  siswa sekolah menengah,  salah  satunya  yaitu Machine Learning. Machine learning telah terbukti menjadi alat yang sangat berguna dalam memproses data besar dan menemukan pola yang tersembunyi, terutama dalam prediksi performa Siswa di dunia Pendidikan.

Lebih lanjut, beberapa penelitian terdahulu telah menunjukkan bahwa penggunaan metode seperti machine learning dan analisis statistik dalam prediksi nilai siswa memberikan hasil yang lebih akurat dibandingkan metode konvensional (Analisis Data untuk Prediksi Akademik). Dengan demikian, prediksi nilai siswa tidak hanya membantu meningkatkan hasil akademik tetapi juga memungkinkan pengelolaan pendidikan yang lebih efisien serta membantu siswa mencapai hasil terbaik mereka.

## Business Understanding
### Problem Statements
Potensi teknologi machine learning untuk menganalisis data akademik belum banyak dimanfaatkan untuk mengidentifikasi pola dan memberikan prediksi yang dapat digunakan oleh pendidik untuk meningkatkan kinerja siswa dan menurunkan tingkat putus sekolah.
### Goals
Memanfaatkan kemampuan machine learning untuk menemukan pola dalam data akademik yang mungkin tersembunyi, guna memberikan informasi yang berguna bagi pengambil kebijakan pendidikan dalam upaya meningkatkan kualitas dan efisiensi sistem pendidikan serta menurunkan angka putus sekolah.
### Solution statements
Menggunakan empat algoritma machine learning, seperti Logistic Regression, Decision Tree, KNN dan Random Forest, untuk membangun model prediksi nilai siswa. Model ini dapat digunakan untuk mengidentifikasi siswa yang berpotensi mengalami kesulitan akademik, terutama dalam mata pelajaran inti seperti matematika dan bahasa. Model akan dievaluasi menggunakan metrik akurasi dan F1-score untuk memastikan prediksi yang tepat dan andal.

## Data Understanding
Dataset diperoleh dari UCI Machine Learning Repository, yang merupakan salah satu platform penyedia dataset untuk data science. Untuk proyek ini, dataset yang saya gunakan adalah Student Performance yang bisa diakses lewat link berikut [https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance). 

Data ini membahas pencapaian siswa dalam pendidikan menengah di dua sekolah di Portugal. Atribut data mencakup nilai siswa, fitur demografis, sosial, dan terkait sekolah, serta dikumpulkan melalui laporan sekolah dan kuesioner. Dua dataset disediakan terkait kinerja dalam dua mata pelajaran berbeda: Matematika (mat) dan Bahasa Portugis (por). Atribut target G3 memiliki korelasi kuat dengan atribut G2 dan G1.

Berikut penjelasan mengenai variabel-variabel pada kolom dataset:
| **Variabel**    | **Keterangan**                                                                                   |
|-----------------|--------------------------------------------------------------------------------------------------|
| school          | Sekolah siswa (biner: 'GP' - Gabriel Pereira atau 'MS' - Mousinho da Silveira)                   |
| sex             | Jenis kelamin siswa (biner: 'F' - perempuan atau 'M' - laki-laki)                                 |
| age             | Usia siswa (numerik: antara 15 hingga 22 tahun)                                                   |
| address         | Tipe alamat rumah siswa (biner: 'U' - urban atau 'R' - rural)                                     |
| famsize         | Ukuran keluarga (biner: 'LE3' - kurang atau sama dengan 3 atau 'GT3' - lebih dari 3)             |
| Pstatus         | Status hubungan orang tua (biner: 'T' - tinggal bersama atau 'A' - terpisah)                      |
| Medu            | Pendidikan ibu (numerik: 0 - tidak ada, 1 - pendidikan dasar (kelas 4), 2 - 5 hingga 9 kelas, 3 - pendidikan menengah, 4 - pendidikan tinggi) |
| Fedu            | Pendidikan ayah (numerik: 0 - tidak ada, 1 - pendidikan dasar (kelas 4), 2 - 5 hingga 9 kelas, 3 - pendidikan menengah, 4 - pendidikan tinggi) |
| Mjob            | Pekerjaan ibu (nominal: 'teacher' - guru, 'health' - terkait kesehatan, 'services' - layanan publik, 'at_home' - di rumah, 'other' - lainnya) |
| Fjob            | Pekerjaan ayah (nominal: 'teacher' - guru, 'health' - terkait kesehatan, 'services' - layanan publik, 'at_home' - di rumah, 'other' - lainnya) |
| reason          | Alasan memilih sekolah ini (nominal: 'home' - dekat rumah, 'reputation' - reputasi sekolah, 'course' - pilihan mata pelajaran, 'other' - lainnya) |
| guardian        | Wali siswa (nominal: 'mother' - ibu, 'father' - ayah, 'other' - lainnya)                         |
| traveltime      | Waktu perjalanan dari rumah ke sekolah (numerik: 1 - <15 menit, 2 - 15 hingga 30 menit, 3 - 30 menit hingga 1 jam, 4 - >1 jam) |
| studytime       | Waktu belajar per minggu (numerik: 1 - <2 jam, 2 - 2 hingga 5 jam, 3 - 5 hingga 10 jam, 4 - >10 jam) |
| failures        | Jumlah kegagalan kelas sebelumnya (numerik: n jika 1<=n<3, jika n>=3 maka 4)                       |
| schoolsup       | Dukungan pendidikan tambahan (biner: yes atau no)                                                 |
| famsup          | Dukungan pendidikan keluarga (biner: yes atau no)                                                |
| paid            | Kelas tambahan berbayar dalam mata pelajaran (Matematika atau Portugis) (biner: yes atau no)     |
| activities      | Kegiatan ekstrakurikuler (biner: yes atau no)                                                   |
| nursery         | Menghadiri taman kanak-kanak (biner: yes atau no)                                                |
| higher          | Ingin melanjutkan pendidikan tinggi (biner: yes atau no)                                          |
| internet        | Akses internet di rumah (biner: yes atau no)                                                     |
| romantic        | Memiliki hubungan romantis (biner: yes atau no)                                                  |
| famrel          | Kualitas hubungan keluarga (numerik: 1 - sangat buruk hingga 5 - sangat baik)                      |
| freetime        | Waktu luang setelah sekolah (numerik: 1 - sangat sedikit hingga 5 - sangat banyak)                |
| goout           | Pergi keluar dengan teman-teman (numerik: 1 - sangat sedikit hingga 5 - sangat banyak)            |
| Dalc            | Konsumsi alkohol pada hari kerja (numerik: 1 - sangat sedikit hingga 5 - sangat banyak)            |
| Walc            | Konsumsi alkohol pada akhir pekan (numerik: 1 - sangat sedikit hingga 5 - sangat banyak)          |
| health          | Status kesehatan saat ini (numerik: 1 - sangat buruk hingga 5 - sangat baik)                      |
| absences        | Jumlah ketidakhadiran sekolah (numerik: dari 0 hingga 93)                                         |
| G1              | Nilai periode pertama (numerik: dari 0 hingga 20)                                                 |
| G2              | Nilai periode kedua (numerik: dari 0 hingga 20)                                                  |
| G3              | Nilai akhir (numerik: dari 0 hingga 20, target output)                                           |

Berikut ini merupakan informasi mengenai jumlah data ,tipe data dan informasi data hilang (missing value) yang terdapat pada dataset.

| **Info Dataset**    | **Missing Value**                                                                               |
|---------------------|----------------------------------------------------------------------------------------------|
| ![info](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/info.png) | ![massing-value](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/missing-value.png) |

Dataset ini terdiri dari 648 data dengan tipe data campuran, yaitu **object** dan **int64**, serta mencakup 33 kolom yang merepresentasikan berbagai atribut yang relevan. Berdasarkan hasil pemeriksaan pada bagian Missing Value, terlihat bahwa tidak ada data yang hilang atau kosong dalam dataset ini. Hal ini menunjukkan bahwa dataset sudah lengkap dan siap untuk dilakukan proses eksplorasi dan analisis lebih lanjut tanpa perlu menangani masalah nilai yang hilang, seperti imputasi atau penghapusan data. Dengan kelengkapan data ini, analisis dapat dilakukan dengan lebih efisien dan fokus pada langkah-langkah preprocessing lainnya.

Untuk mempermudah proses analisis diperlukan beberapa visualisasi data, seperti berikut ini :
- Melihat sebaran data pada seluruh variabel dan hubungan pada setiap variabel
![sebaran-data](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/sebaran-data.png)
- ***sns.catplot***, untuk melihat distribusi data.
![absens](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/absences.png)
- ***sns.pairplot***, untuk mengamati hubungan antara fitur numerik.
![pairplot](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/pairplot.png)
- ***sns.heatmap***, untuk mengamati hubungan korelasi antar fitur.
![matrik-korelasi](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/matrik-korelasi.png)

## Data Preparation 
Data preparation merupakan tahapan penting dalam proses pengembangan model machine learning. Ini adalah tahap di mana kita melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahapan yang umum dilakukan pada data preparation, antara lain:
- Menghapus Data dengan Nilai G3 Kurang dari 1 
`df.drop(df[df['G3'] < 1].index, inplace=True)` : kode disamping digunakan untuk menghapus semua data yang memiliki nilai kolom G3 kurang dari 1. Hal ini bertujuan untuk memastikan bahwa hanya data dengan nilai G3 yang valid dan relevan yang akan digunakan dalam analisis. 
- Mengonversi Variabel Kategorikal ke Bentuk ***One-Hot Encoding***
One-Hot Encoding adalah teknik dalam pra-pemrosesan data yang digunakan untuk mengubah data kategori menjadi bentuk numerik agar dapat digunakan oleh model machine learning.
`df_ohe = pd.get_dummies(df, drop_first=True)` kode ini digunakan untuk mengubah variabel kategorikal dalam dataset menjadi format one-hot encoding. Dengan drop_first=True, proses ini akan menghindari masalah multikolinearitas dengan membuang salah satu kategori referensi dari setiap variabel kategorikal. Hasilnya adalah dataset dengan semua fitur yang dapat digunakan dalam analisis numerik atau model machine learning.
- Menentukan Batas Minimum Korelasi (***Threshold***)
Variabel **THRESHOLD** ditetapkan dengan nilai **0.13**, yang digunakan untuk menyaring kolom-kolom dalam dataset berdasarkan nilai korelasi absolutnya terhadap kolom G3. Kolom dengan korelasi absolut kurang dari batas ini dianggap kurang relevan dan akan dihapus dari dataset.
- Menghapus Fitur dengan Korelasi Rendah
```
for key, value in G3_corr.items():
  if abs(value) < THRESHOLD:
    df_ohe_after_drop_features.drop(columns= key, inplace=True)
````
Dengan menggunakan loop, setiap kolom yang memiliki nilai korelasi absolut lebih kecil dari THRESHOLD dihapus dari dataset, sehingga dataset hanya menyisakan kolom-kolom yang memiliki hubungan yang signifikan dengan target G3.
- Membagi Dataset menjadi Fitur (X) dan Target (y)
```
X = df_ohe_after_drop_features.drop('G3', axis=1)
y = df_ohe_after_drop_features['G3']
```
Pada proses ini, data dipisahkan menjadi dua bagian utama, yaitu X dan y. X merupakan kumpulan fitur independen yang berisi semua kolom dalam dataset kecuali kolom G3. Penghapusan kolom G3 dilakukan menggunakan perintah drop('G3', axis=1), sehingga data yang tersisa pada X hanya mencakup fitur-fitur yang akan digunakan sebagai input untuk model machine learning. Sementara itu, y berisi kolom G3, yang merupakan variabel target atau nilai yang ingin diprediksi oleh model. Pemisahan ini penting untuk memastikan bahwa model dilatih menggunakan data fitur independen (X) untuk mempelajari hubungan atau pola yang berkaitan dengan variabel target (y).

- ***Train Test Split*** : Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. 
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
```
Kode ini membagi dataset menjadi dua bagian, yaitu 80% data latih dan 20% data uji, dengan data diacak terlebih dahulu. Hal ini dilakukan agar model dapat dilatih pada data yang representatif dan diuji pada data yang belum pernah dilihat, sehingga memberikan evaluasi performa yang lebih akurat.

hasil dari kode diatas dapat dilihat pada data berikut : 
`Total of sample in whole dataset: 634`
`Total of sample in train dataset: 507`
`Total of sample in test dataset: 127`
- Menerapkan teknik Standarisasi : Proses standarisasi dapat membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.
![standarisasi](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/standarisasi.png)

## Modeling
Pada tahap ini, kita akan mengembangkan model machine learning dengan empat algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Keempat algoritma yang akan kita gunakan, antara lain:
- **Regresi logistik** adalah salah satu algortima pembelajaran mesin yang handal digunakan untuk klasifikasi data dengan target bertipe kategori ([Saleh, M., & Chamidy, T., 2024](https://doi.org/10.36040/jati.v8i3.9696))
```
model1 = LogisticRegression(max_iter=200)
```
pada Logistic Regression menggunakan parameter `max_iter=200`. Parameter tersebut digunakan untuk menentukan jumlah iterasi maksimum yang diperbolehkan bagi algoritma pengoptimalan untuk mencapai konvergensi. Nilai 200 digunakan untuk memastikan model memiliki cukup iterasi guna menemukan solusi optimal.

- **Decision Tree Classifier**
Decision Tree adalah algoritma machine learning yang sering digunakan dalam tugas klasifikasi dan regresi. Struktur dari algoritma ini mirip dengan bentuk pohon dengan setiap cabang mewakili keputusan atau percabangan dari data berdasarkan fitur-fitur yang ada.
```
model2 = DecisionTreeClassifier()
```
Parameter pada model DecisionTreeClassifier menggunakan setingan default, yang berarti: 
  - `criterion='gini'`: Model menggunakan Gini Impurity untuk mengukur kualitas split.
  - `splitter='best'`: Model mencari split terbaik pada setiap langkah.
Parameter lain seperti kedalaman pohon atau jumlah sampel minimum juga menggunakan nilai bawaan yang ditentukan oleh pustaka scikit-learn.
- K-Nearest Neighbors (KNN) adalah algoritma machine learning yang sederhana dan mudah dipahami untuk klasifikasi dan regresi. Algoritma ini bekerja dengan menemukan k tetangga terdekat dari data baru dan kemudian menggunakan kategori atau nilai rata-rata dari tetangga tersebut untuk memprediksi kategori atau nilai data baru ([Mardiyyah, N.W., Rahaningsih, N. and Ali, I., 2024.](https://doi.org/10.36040/jati.v8i2.9010)).
Dalam algoritma KNN, beberapa parameter utama perlu diatur untuk mengoptimalkan performa model. Inilah beberapa di antaranya.
  - Jumlah Tetangga (K)
  - Metric Jarak
  - Bobot (Weights)
  - Panjang Jarak (Distance Metric Parameters)
  - Normalisasi Data
  - Algoritma Pencarian Tetangga

```
model3 = KNeighborsClassifier()
```
Parameter pada model KNeighborsClassifier juga menggunakan diseting secara default. Ini berarti:
  - `n_neighbors=5`: Model mempertimbangkan 5 tetangga terdekat untuk menentukan klasifikasi.
  - `metric='minkowski'`: Model menggunakan metrik Minkowski untuk menghitung jarak.
  - `weights='uniform'`: Semua tetangga memiliki bobot yang sama dalam prediksi.

- Random Forest adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. ([Husin, N., 2023.](https://pdfs.semanticscholar.org/f048/ea7a08cfca1a0a705eb5fd57efd7ba798fa2.pdf)).
Dalam algoritma Random Forest, beberapa parameter utama perlu diatur untuk mengoptimalkan performa model. Inilah beberapa di antaranya.
  - n_estimators
  - Maksimal Kedalaman (max_depth)
  - Jumlah Minimum Sampel untuk Split (min_samples_split)
  - Jumlah Minimum Sampel untuk Daun (min_samples_leaf)
  - Jumlah Maksimum Fitur untuk Pembagian (max_features)
  - bootstrap
  - random_state
pada model yang kita gunakan seperti pada kode berikut :
```
model4 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=123, n_jobs=-1, class_weight='balanced')
```
keterangan parameter dapat dilihat dibawah ini :
  - `n_estimators=100`: Menentukan jumlah pohon keputusan yang akan dibuat acak. Dalam penelitian ini, 100 pohon digunakan untuk meningkatkan stabilitas dan akurasi model.
  - `max_depth=10`: Menentukan kedalaman maksimum pohon keputusan. Pembatasan ini membantu menghindari overfitting pada data.
  - `random_state=123`: Menetapkan nilai acak untuk memastikan replikasi hasil, sehingga eksperimen dapat diuji ulang dengan hasil yang konsisten.
  - `n_jobs=-1`: Mengizinkan model untuk menggunakan semua inti prosesor yang tersedia untuk mempercepat pelatihan.
  - `class_weight='balanced'`: Menyesuaikan bobot kelas berdasarkan distribusi data. Parameter ini digunakan untuk menangani ketidakseimbangan data, sehingga kelas minoritas memiliki kontribusi yang lebih signifikan dalam pelatihan.
  
## Evaluasi
Pada bagian ini, evaluasi dilakukan untuk mengukur performa model yang telah dibangun menggunakan metrik evaluasi seperti akurasi dan *F1-score*. 
- **Confusion matrix atau matriks kebingungan** adalah alat yang digunakan untuk menggambarkan kinerja model klasifikasi pada data uji yang sudah diketahui hasil sebenarnya. Matriks kebingungan digunakan untuk menganalisis kinerja model klasifikasi dengan membandingkan hasil prediksi model terhadap data aktual. Matriks ini memberikan gambaran mengenai prediksi benar (true positive dan true negative) serta kesalahan prediksi (false positive dan false negative) dari setiap model.
![confusion matrix](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/confusion-matrix.png)
Untuk melihat hasil pelatihan dari masing-masing model klasifikasi dengan menggunakan akurasi pada nilai yang dihasilkan pada setiap model, nilai akurasi menggunakan library dari sklearn. Selainmelihat nilai akurasi pada proyek ini melakukan visualisasi hasil pelatihan dengan confusion matrix. Berikut merupakan hasil akurasi pada setiap model:
- **Evaluasi Hasil Model - Pengujuan 1**
Berikut adalah hasil akurasi dan matriks kebingungan dari pengujian pertama menggunakan keempat algoritma:
  - **Akurasi dan F1-Score - Pengujian 1**
![](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/hasil-pengujian-1.png)
  - **Matriks Kebingungan - Pengujian 1**
![](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/confusion-matrix-1.png)
- **Evaluasi Hasil Model - Pengujuan 2**
Hasil pengujian kedua menunjukkan performa algoritma yang serupa dengan pengujian pertama:
  - **Akurasi dan F1-Score - Pengujian 2**
![](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/hasil-pengujian-2.png)
  - **Matriks Kebingungan - Pengujian 1**
![](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/confusion-matrix-2.png)

Dari hasil evaluasi dapat diidentifikasi sebagai berikut :
1. **Dampak Model terhadap Problem Statement**
Model yang dibangun berhasil mengidentifikasi pola dalam data akademik dengan baik, sebagaimana ditunjukkan oleh nilai akurasi yang tinggi pada Logistic Regression dan Random Forest. Hal ini menunjukkan potensi implementasi machine learning dalam membantu pendidik mengidentifikasi siswa yang membutuhkan perhatian lebih. Dengan demikian, penelitian ini berhasil menjawab problem statement tentang rendahnya pemanfaatan teknologi untuk memahami data akademik.

2. **Capaian terhadap Goals**
Hasil evaluasi menunjukkan bahwa model dapat digunakan untuk memberikan informasi prediktif yang berguna bagi pengambil kebijakan pendidikan. Sebagai contoh, model Random Forest dengan akurasi tertinggi dapat digunakan untuk memprioritaskan sumber daya pendidikan pada siswa yang memiliki risiko kesulitan akademik. Dengan ini, goals untuk meningkatkan kualitas pendidikan dan mengurangi angka putus sekolah mulai tercapai.

3. **Relevansi dengan Solution Statement**
Keempat algoritma yang digunakan berhasil memberikan prediksi dengan tingkat keandalan yang baik. Logistic Regression memberikan solusi yang sederhana dan mudah diinterpretasikan, sementara Random Forest memberikan prediksi yang lebih kompleks dan akurat. Meskipun demikian, terdapat perbedaan performa pada pengujian data tertentu, sehingga pendekatan model ensemble seperti Random Forest menjadi pilihan yang lebih relevan untuk implementasi praktis.

4. **Keterbatasan dan Implikasi**
Model dengan akurasi tinggi belum tentu optimal tanpa mempertimbangkan aspek F1-Score, terutama jika terdapat ketidakseimbangan kelas dalam data. Oleh karena itu, evaluasi lebih lanjut pada data baru diperlukan untuk memastikan bahwa solusi ini dapat diimplementasikan secara konsisten dalam konteks nyata.

Secara keseluruhan, evaluasi model menunjukkan hasil yang positif dalam menjawab problem statements dan mendukung pencapaian goals penelitian. Model yang dibangun tidak hanya mampu memprediksi nilai siswa dengan akurat, tetapi juga memberikan wawasan penting yang dapat digunakan untuk meningkatkan kualitas sistem pendidikan. Selanjutnya, model ini memiliki potensi untuk diimplementasikan sebagai alat bantu dalam sistem pendidikan berbasis data.
## Referensi
- BRITO, A. ; TEIXEIRA, J., eds. lit. – “Proceedings of 5th Annual Future Business Technology Conference, Porto, 2008”. [S.l. : EUROSIS, 2008]. ISBN 978-9077381-39-7. p. 5-12.
- Saleh, M. and Chamidy, T., 2024. ANALISIS FAKTOR-FAKTOR YANG MEMPENGARUHI TINGKAT KEPUASAN SISWA MTS SURYA BUANA MENGGUNAKAN METODE REGRESI LOGISTIK. JATI (Jurnal Mahasiswa Teknik Informatika), 8(3), pp.3463-3470.
- Mardiyyah, N.W., Rahaningsih, N. and Ali, I., 2024. PENERAPAN DATA MINING MENGGUNAKAN ALGORITMA K-NEAREST NEIGHBOR PADA PREDIKSI PEMBERIAN KREDIT DI SEKTOR FINANSIAL. JATI (Jurnal Mahasiswa Teknik Informatika), 8(2), pp.1491-1499.
- Husin, N., 2023. Komparasi Algoritma Random Forest, Naïve Bayes, dan Bert Untuk Multi-Class Classification Pada Artikel Cable News Network (CNN). J. Esensi Infokom J. Esensi Sist. Inf.
