Saya membangun language model dari nol, lalu membuat alat untuk melihat isi kepalanya.

Glassbox adalah transformer 2,6 juta parameter yang saya latih di CPU laptop dalam 3 jam. Semuanya berjalan di browser: tanpa server, tanpa API key.

Lalu saya bedah:
→ Sparse autoencoder memecah state internalnya menjadi 2.048 fitur yang bisa dibaca.
→ Steering dengan satu fitur ("ball", "dog", "cat", ...) membuat 83% cerita menyebut konsep itu, dibanding 9% dengan arah acak berukuran sama.
→ Pengaman nama membaca fitur-fitur itu sebelum tiap token dipilih. Digabung dengan regex, ia menangkap 71% nama yang tidak pernah dipakai untuk melatihnya; regex saja hanya 52%. Harganya: sekitar 2 dari 100 token lain ikut tertahan karena salah deteksi.

Yang paling saya jaga adalah disiplinnya:
• setiap angka dibuat oleh skrip dari hasil eksperimen, dan CI mengecek halaman masih cocok
• setiap klaim diukur terhadap pembanding
• aturan pemilihan ditulis sebelum hasilnya keluar
• hasil negatif tetap dilaporkan: induction head tidak terbentuk, dan pengaman fitur sendirian tidak mengalahkan regex

Dibangun bersama Claude Code sebagai AI pair programmer, berdasarkan brief proyek tertulis. Tujuan, keputusan, dan review hasil ada di tangan saya.

Coba demonya: https://muhimranhidayathamzh.github.io/glassbox/
Studi kasus (EN): https://muhimranhidayathamzh.github.io/glassbox/case-study.html
Kode: https://github.com/muhimranhidayathamzh/glassbox

#MachineLearning #Interpretability #LLM #AIEngineering
