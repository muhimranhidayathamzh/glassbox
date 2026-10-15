# Glassbox — Brief Proyek

LLM kecil yang dibangun dari nol, ditambah alat interaktif di browser untuk melihat, mengendalikan,
dan melacak cara berpikirnya (mechanistic interpretability), plus satu fitur produk yang lahir dari temuan itu.

Tesis: *"Saya membangun LLM dari nol, membedah cara berpikirnya dengan teknik terobosan 2026,
lalu mengubah temuan itu menjadi fitur produk yang terukur."*

Tujuan akhir: repo portofolio AI Engineer + demo statis di GitHub Pages yang bisa dibuka siapa pun,
tanpa server dan tanpa API key.

## Prinsip yang tidak boleh dilanggar

1. **Jangan pernah mengarang hasil.** Semua angka di README, demo, dan slide berasal dari skrip yang bisa dijalankan ulang.
2. **Ilustrasi diberi label.** Contoh yang bukan output asli model wajib ditandai "ilustrasi".
3. **Selalu ada pembanding.** Setiap klaim diukur terhadap baseline (arah acak, regex, classifier sederhana).
4. **Hasil negatif dilaporkan.** Kalau fitur tidak bersih, steering tidak berefek, atau induction head tidak muncul, tulis apa adanya.
5. **Deterministik.** Seed tetap untuk training, SAE, dan evaluasi.

## Tiga lapisan

| Lapisan | Isi | Bukti untuk |
|---|---|---|
| 1. Dibangun dari nol | Tokenizer BPE, transformer decoder-only, training loop, mesin inferensi JavaScript | Fondasi |
| 2. Mikroskop | Sparse autoencoder (SAE), steering, attribution/sirkuit sederhana, replay training | Riset terkini |
| 3. Pengaman pembaca pikiran | Mencegah model menulis nama orang dengan membaca aktivasi fitur sebelum token dipilih | Produk |

## Struktur repo yang diusulkan

```
glassbox/
├── python/glassbox/
│   ├── tokenizer.py      # BPE dari nol (train, encode, decode, save/load)
│   ├── model.py          # transformer: attention, MLP, residual, LayerNorm (hanya torch.nn primitif)
│   ├── train.py          # training loop + checkpoint berjarak logaritmik untuk replay
│   ├── sae.py            # sparse autoencoder pada residual stream
│   ├── features.py       # contoh aktivasi teratas per fitur, label otomatis sederhana
│   ├── steer.py          # tambah arah decoder fitur × alpha ke residual stream
│   ├── attribution.py    # direct logit attribution fitur + ablasi
│   ├── induction.py      # skor induction head per checkpoint
│   ├── guard.py          # deteksi nama berbasis fitur + baseline regex & classifier
│   ├── export.py         # ekspor bobot (float16/int8) + manifest JSON untuk web
│   └── eval/             # semua benchmark; tulis hasil ke results/*.json dan *.md
├── web/                  # demo statis (vanilla JS/TS + Vite boleh), tanpa backend
│   ├── engine/           # forward pass JS, KV cache, sampling
│   └── ui/               # mikroskop, slider steering, graf sirkuit, replay, pengaman
├── tests/                # pytest + tes paritas Node vs PyTorch
├── results/              # hasil benchmark (dibuat skrip, bukan ditulis tangan)
└── .github/workflows/    # CI: lint, tes, paritas, deploy Pages
```

## Spesifikasi awal (boleh disesuaikan setelah eksperimen)

- **Data:** cerita pendek berbahasa Inggris yang sederhana (misalnya dataset jenis TinyStories) agar model kecil bisa menghasilkan kalimat koheren. Cek lisensi dataset. Opsi lanjutan: teks bahasa Indonesia.
- **Model:** vocab 2.000–4.000, d_model 128–256, 4 layer, 4 head, konteks 256 token, target 1–5 juta parameter. Harus bisa dilatih di CPU laptop dalam hitungan jam.
- **SAE:** pada residual stream layer tengah, ukuran kamus 8–16× d_model, L1 atau TopK. Laporkan:
  - variance explained (rekonstruksi)
  - L0: rata-rata fitur aktif per token
  - persentase fitur mati
  - kenaikan loss model saat aktivasi diganti hasil rekonstruksi SAE
- **Steering:** bandingkan efek arah fitur dengan arah acak berukuran sama. Ukur perubahan frekuensi token terkait konsep.
- **Attribution:** versi sederhana. Direct logit attribution fitur ke token output, divalidasi dengan ablasi.
- **Induction head:** skor prefix-matching pada urutan token acak yang diulang, dihitung per checkpoint. Tandai apakah ada penurunan loss tajam yang bertepatan.
- **Pengaman:** dataset cerita dengan nama berlabel (bisa dibuat secara sintetis). Fitur "nama" melewati ambang → token diblokir sebelum keluar. Baseline: regex kata berhuruf kapital + daftar nama, dan logistic regression. Metrik: precision, recall, F1, latensi tambahan per token.
- **Web:** tes paritas wajib. Output logit JS vs PyTorch untuk input yang sama harus berada di bawah ambang selisih maksimum yang ditetapkan.

## Milestone (selesaikan berurutan, masing-masing dengan definisi selesai)

1. **Tokenizer + model + training.** Loss turun, sampel teks koheren, checkpoint tersimpan, tes unit lulus.
2. **Inferensi web.** Model berjalan di browser, tes paritas lulus, generate teks interaktif.
3. **SAE + mikroskop.** Metrik SAE dilaporkan, UI pencarian fitur dengan contoh aktivasi teratas.
4. **Steering + attribution.** Slider di UI, benchmark vs arah acak, graf sirkuit untuk kata yang diklik.
5. **Replay training.** Slider checkpoint, kurva loss, skor induction head per checkpoint.
6. **Pengaman.** Benchmark vs regex dan classifier, demo blokir nama di UI.
7. **Polish.** README berisi hasil asli, GIF demo, CI hijau, deploy GitHub Pages.

## Kendala

- Semua harus bisa dijalankan di laptop tanpa GPU dan tanpa API key.
- Demo final berupa situs statis; bobot model cukup kecil untuk dimuat di browser.
- Dependensi Python minimal: torch, numpy, pytest. Hindari library interpretability siap pakai untuk komponen inti, karena tujuan proyek ini adalah membangunnya sendiri.

## Opsi skripsi (catatan)

Kalau dijadikan skripsi, persempit ke satu pertanyaan penelitian:
- **Utama:** apakah SAE menemukan fitur yang mencerminkan morfologi bahasa Indonesia (imbuhan me-, di-, -kan) pada model kecil berbahasa Indonesia?
- **Cadangan:** perbandingan deteksi nama berbasis fitur SAE vs regex vs classifier (precision, recall, latensi).
