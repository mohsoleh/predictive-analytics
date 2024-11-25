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
Dataset diperoleh dari UCI Machine Learning Repository, yang merupakan salah satu platform penyedia dataset untuk data science. Untuk proyek ini, dataset yang saya gunakan adalah Student Performance yang bisa diakses [disini](https://archive.ics.uci.edu/dataset/320/student+performance). Berikut penjelasan mengenai variabel-variabel pada kolom dataset:
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
![info](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/info.png) | ![massing-value](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/missing-value.png)

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
Beberapa langkah yang perlu kita dilakukan sebelum melakukan tahapan Data Preparation:
- Melakukan pemeriksaan terhadap nilai yang hilang(missing value) pada dataset
  ![massing-value](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/missing-value.png)
- Memeriksa outlier dengan metode IQR.
  - ***Outliers*** adalah sampel yang nilainya sangat jauh dari cakupan umum data utama. Ia adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya. Kita dapat menangani outliers dengan teknik IQR method.
  - ***IQR*** adalah singkatan dari Inter Quartile Range. Untuk memahami apa itu IQR, dibutuhkan pemahaman terhadap konsep kuartil. Kuartil dari suatu populasi adalah tiga nilai yang membagi distribusi data menjadi empat sebaran. Seperempat dari data berada di bawah kuartil pertama (Q1), setengah dari data berada di bawah kuartil kedua (Q2), dan tiga perempat dari data berada di kuartil ketiga (Q3). Dengan demikian interquartile range atau IQR = Q3 - Q1. Kita dapat menggunakan metode IQR untuk mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier.
- ***Train Test Split*** : Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. 
- Menerapkan teknik Standarisasi : Proses standarisasi dapat membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.
![standarisasi](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/standarisasi.png)

## Modeling
- **Regresi logistik** adalah salah satu algortima pembelajaran mesin yang handal digunakan untuk klasifikasi data dengan target bertipe kategori ([Saleh, M., & Chamidy, T., 2024](https://doi.org/10.36040/jati.v8i3.9696))
- **Decision Tree Classifier**
- K-Nearest Neighbors (KNN) adalah algoritma machine learning yang sederhana dan mudah dipahami untuk klasifikasi dan regresi. Algoritma ini bekerja dengan menemukan k tetangga terdekat dari data baru dan kemudian menggunakan kategori atau nilai rata-rata dari tetangga tersebut untuk memprediksi kategori atau nilai data baru ([Mardiyyah, N.W., Rahaningsih, N. and Ali, I., 2024.](https://doi.org/10.36040/jati.v8i2.9010)).
- Random Forest adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. ([Husin, N., 2023.](https://pdfs.semanticscholar.org/f048/ea7a08cfca1a0a705eb5fd57efd7ba798fa2.pdf)).
## Evaluasi
- **Confusion matrix atau matriks kebingungan** adalah alat yang digunakan untuk menggambarkan kinerja model klasifikasi pada data uji yang sudah diketahui hasil sebenarnya. Confusion matrix merupakan cara kita mencatat poin benar dan poin salah tersebut. Di dalam matriks ini, kita tulis semua kemungkinan jawaban yang benar dan jawaban yang salah.
![confusion matrix](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/confusion-matrix.png)
Untuk melihat hasil pelatihan dari masing-masing model klasifikasi dengan menggunakan akurasi pada nilai yang dihasilkan pada setiap model, nilai akurasi menggunakan library dari sklearn. Selainmelihat nilai akurasi pada proyek ini melakukan visualisasi hasil pelatihan dengan confusion matrix. Berikut merupakan hasil akurasi pada setiap model:
- **Evaluasi Hasil Model - Pengujuan 1**
![](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/hasil-pengujian-1.png)
- Berikut merupakan hasil dari confusion matrix - Pengujian 1
![](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/confusion-matrix-1.png)
- **Evaluasi Hasil Model - Pengujuan 2**
![](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/hasil-pengujian-2.png)
- Berikut merupakan hasil dari confusion matrix - Pengujian 2
![](https://raw.githubusercontent.com/mohsoleh/predictive-analytics/refs/heads/main/img/confusion-matrix-2.png)
