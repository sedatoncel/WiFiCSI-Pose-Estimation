"""
=============================================================
 MMFi — WiFi CSI -> 3D İskelet : EĞİTİM + VALIDATION + GRAFİKLER
=============================================================
 Kurulum:
   train      = E01 + E02
   validation = E04   (her epoch izlenir, en iyi model buna göre seçilir)
   test       = E03   (AYRI scriptte, sona saklanır)

 VALIDATION YÖNTEMİ (rapora):
   Cross-scene (ortamlar-arası) protokol kullanılır: model E01+E02
   odalarında eğitilir, hiç görmediği E04 odasında her epoch sonunda
   değerlendirilir. Validation metriği olarak gerçek MPJPE (eklem başına
   Öklid mesafesi, mm) ve eksen-başına MAE (X/Y/Z) kullanılır. Bu metrik
   doğrudan tahmin kalitesini ölçer ve modelin öğrenip öğrenmediğini
   "ortalama poz" tabanıyla kıyaslar. En düşük validation MPJPE'sini veren
   epoch'un ağırlıkları "en iyi model" olarak saklanır (model seçimi).
   Bu protokol, sistemin GÖRÜLMEMİŞ bir ortama genelleyip genellemediğini
   ölçer — WiFi sensing'in temel zorluğu budur.

 Üretilen grafikler (./grafikler/ klasörüne PNG):
   1) loss_egrileri.png      - train & val L1 kaybı / epoch
   2) val_mpjpe_egrisi.png   - validation MPJPE (mm) / epoch
   3) eksen_mae_bar.png      - final model X/Y/Z MAE
   4) model_vs_taban.png     - in-domain (train) vs validation; model vs taban
   5) pck_egrisi.png         - PCK: eşik (cm) vs doğru eklem yüzdesi
=============================================================
"""

import os
import copy
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")          # ekran olmadan PNG kaydı için
import matplotlib.pyplot as plt

from mmfi import make_dataset
from file3_model import CSITransformerEncoder


# ----------------------- yardımcılar -----------------------
def hazirla(data, device):
    wifi = data['input_wifi-csi'].to(device).float()
    iskelet = data['output'].to(device).float()           # (B,17,3)
    B = wifi.size(0)
    if wifi.is_complex():
        wifi = torch.view_as_real(wifi)
    wifi = wifi.permute(0, 2, 1, 3, *range(4, wifi.dim()))
    wifi_tokenlari = wifi.reshape(B, 114, -1)              # (B,114,30)
    gercek_root = iskelet[:, 0, :]
    merkezli = iskelet - iskelet[:, 0:1, :]
    return wifi_tokenlari, iskelet, gercek_root, merkezli


@torch.no_grad()
def degerlendir(model, loader, device, pck_esikler_m=None):
    """MPJPE(mm), eksen-MAE[X,Y,Z](mm), root(mm), val L1, ve istenirse PCK."""
    model.eval()
    toplam_eklem = 0
    toplam_ornek = 0
    mpjpe_sum = 0.0
    eksen_sum = torch.zeros(3, device=device)
    root_err_sum = 0.0
    l1_sum = 0.0
    # PCK için eşik altındaki eklem sayıları
    if pck_esikler_m is not None:
        pck_dogru = torch.zeros(len(pck_esikler_m), device=device)

    for data in loader:
        wifi_tokenlari, _, gercek_root, merkezli = hazirla(data, device)
        B = wifi_tokenlari.size(0)
        pred_root, pred_kp, _ = model(wifi_tokenlari)
        pred_kp = pred_kp.view(B, 17, 3)

        per_joint = torch.norm(pred_kp - merkezli, dim=-1)        # (B,17) metre
        mpjpe_sum += per_joint.sum().item()
        eksen_sum += (pred_kp - merkezli).abs().sum(dim=(0, 1))
        root_err_sum += torch.norm(pred_root - gercek_root, dim=-1).sum().item()
        l1_sum += (pred_kp - merkezli).abs().mean().item() * B    # ~batch L1, ornek-agirlikli

        if pck_esikler_m is not None:
            for k, esik in enumerate(pck_esikler_m):
                pck_dogru[k] += (per_joint < esik).sum()

        toplam_eklem += B * 17
        toplam_ornek += B

    mpjpe_mm = (mpjpe_sum / toplam_eklem) * 1000.0
    eksen_mm = (eksen_sum / toplam_eklem * 1000.0).cpu().tolist()
    root_mm = (root_err_sum / toplam_ornek) * 1000.0
    val_l1 = l1_sum / toplam_ornek
    if pck_esikler_m is not None:
        pck = (pck_dogru / toplam_eklem * 100.0).cpu().tolist()
        return mpjpe_mm, eksen_mm, root_mm, val_l1, pck
    return mpjpe_mm, eksen_mm, root_mm, val_l1


@torch.no_grad()
def taban_eksen_mae(train_loader, hedef_loader, device, max_batch=400):
    """Eğitim ortalaması (sabit poz) -> hedef ortamda eksen-başına MAE [X,Y,Z] mm."""
    toplam = torch.zeros(17, 3, device=device)
    n = 0
    for i, data in enumerate(train_loader):
        if i >= max_batch:
            break
        _, _, _, merkezli = hazirla(data, device)
        toplam += merkezli.sum(dim=0)
        n += merkezli.size(0)
    sabit = toplam / n
    eksen_sum = torch.zeros(3, device=device)
    te = 0
    for i, data in enumerate(hedef_loader):
        if i >= max_batch:
            break
        _, _, _, merkezli = hazirla(data, device)
        eksen_sum += (sabit.unsqueeze(0) - merkezli).abs().sum(dim=(0, 1))
        te += merkezli.size(0) * 17
    return (eksen_sum / te * 1000.0).cpu().tolist()


if __name__ == '__main__':
    dataset_root = r"C:\Wifi-Sens data\MMFi_Dataset"
    os.makedirs("grafikler", exist_ok=True)

    with open('config.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    # validation = E04 (config zaten E04 veriyor)
    print("Veriler yükleniyor...")
    train_dataset, val_dataset = make_dataset(dataset_root, config)   # train=E01+E02, val=E04
    print(f"Train (E01+E02): {len(train_dataset)} | Validation (E04): {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)
    # taban hesabı için karıştırılmamış train görünümü
    train_eval_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                   num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    model = CSITransformerEncoder(raw_token_dim=30).to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    epochs = 30
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    LAMBDA_ROOT = 0.1
    PCK_ESIKLER = [0.025, 0.05, 0.075, 0.10, 0.15, 0.20]   # metre

    # kayıt için geçmiş
    hist_train_l1, hist_val_l1, hist_val_mpjpe = [], [], []
    best_val_mpjpe = float('inf')

    print("\nEğitim başlıyor...")
    for epoch in range(epochs):
        model.train()
        toplam_kp = 0.0
        for step, data in enumerate(train_loader):
            wifi_tokenlari, _, gercek_root, merkezli = hazirla(data, device)
            B = wifi_tokenlari.size(0)

            optimizer.zero_grad()
            pred_root, pred_kp, _ = model(wifi_tokenlari)
            pred_kp = pred_kp.view(B, 17, 3)

            loss_kp = criterion(pred_kp, merkezli)
            loss_root = criterion(pred_root, gercek_root)
            loss = loss_kp + LAMBDA_ROOT * loss_root

            loss.backward()
            optimizer.step()
            toplam_kp += loss_kp.item()

            if step % 1000 == 0 and step > 0:
                print(f"  Tur [{epoch+1}/{epochs}] Adım [{step}/{len(train_loader)}] kp: {loss_kp.item():.4f}")

        scheduler.step()
        train_l1 = toplam_kp / len(train_loader)

        # --- VALIDATION (E04) ---
        val_mpjpe, val_eksen, val_root, val_l1 = degerlendir(model, val_loader, device)

        hist_train_l1.append(train_l1)
        hist_val_l1.append(val_l1)
        hist_val_mpjpe.append(val_mpjpe)

        print(f"*** Tur [{epoch+1}/{epochs}] | train L1: {train_l1:.4f} | "
              f"val L1: {val_l1:.4f} | val MPJPE: {val_mpjpe:.1f} mm | "
              f"X/Y/Z: {val_eksen[0]:.1f}/{val_eksen[1]:.1f}/{val_eksen[2]:.1f}")

        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            torch.save(model.state_dict(), "wifi_iskelet_modeli_best.pth")
            print(f"    -> en iyi model kaydedildi (val MPJPE {best_val_mpjpe:.1f} mm)")

    torch.save(model.state_dict(), "wifi_iskelet_modeli_son.pth")
    print(f"\nEğitim bitti. En iyi val MPJPE: {best_val_mpjpe:.1f} mm")

    # ===================== GRAFİKLER =====================
    print("\nGrafikler üretiliyor...")
    model.load_state_dict(torch.load("wifi_iskelet_modeli_best.pth",
                                     map_location=device, weights_only=True))
    epoch_ekseni = list(range(1, epochs + 1))

    # 1) Loss eğrileri
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_ekseni, hist_train_l1, 'o-', label='Train L1 (E01+E02)')
    plt.plot(epoch_ekseni, hist_val_l1, 's-', label='Validation L1 (E04)')
    plt.xlabel('Epoch'); plt.ylabel('L1 Loss (m)')
    plt.title('Öğrenme Eğrisi — Train vs Validation Loss')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("grafikler/loss_egrileri.png", dpi=150); plt.close()

    # 2) Validation MPJPE eğrisi
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_ekseni, hist_val_mpjpe, 'd-', color='tab:red', label='Validation MPJPE (E04)')
    plt.axhline(best_val_mpjpe, ls='--', color='gray', alpha=0.7,
                label=f'En iyi: {best_val_mpjpe:.1f} mm')
    plt.xlabel('Epoch'); plt.ylabel('MPJPE (mm)')
    plt.title('Validation MPJPE / Epoch (E04 — görülmemiş ortam)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("grafikler/val_mpjpe_egrisi.png", dpi=150); plt.close()

    # 3) Final eksen-başına MAE (X/Y/Z)
    val_mpjpe, val_eksen, _, _, val_pck = degerlendir(
        model, val_loader, device, pck_esikler_m=PCK_ESIKLER)
    plt.figure(figsize=(7, 5))
    barlar = plt.bar(['X', 'Y (derinlik)', 'Z'], val_eksen,
                     color=['tab:blue', 'tab:red', 'tab:green'])
    for b, v in zip(barlar, val_eksen):
        plt.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.1f}', ha='center')
    plt.ylabel('MAE (mm)')
    plt.title('Eksen-Başına Hata (Validation E04) — Y en zor eksen')
    plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig("grafikler/eksen_mae_bar.png", dpi=150); plt.close()

    # 4) Model vs Taban — in-domain (train) vs validation
    tr_taban = taban_eksen_mae(train_eval_loader, train_eval_loader, device)
    val_taban = taban_eksen_mae(train_eval_loader, val_loader, device)
    _, tr_model_eksen, _, _ = degerlendir(model, train_eval_loader, device)

    etiketler = ['X', 'Y', 'Z']
    x = np.arange(3); w = 0.2
    plt.figure(figsize=(9, 5))
    plt.bar(x - 1.5*w, tr_model_eksen, w, label='Model — Train (in-domain)')
    plt.bar(x - 0.5*w, tr_taban, w, label='Taban — Train')
    plt.bar(x + 0.5*w, val_eksen, w, label='Model — Val (E04)')
    plt.bar(x + 1.5*w, val_taban, w, label='Taban — Val (E04)')
    plt.xticks(x, etiketler); plt.ylabel('MAE (mm)')
    plt.title('Model vs Taban: gördüğü ortam çalışıyor, görmediği ortam tabana düşüyor')
    plt.legend(fontsize=8); plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig("grafikler/model_vs_taban.png", dpi=150); plt.close()

    # 5) PCK eğrisi
    plt.figure(figsize=(8, 5))
    esik_cm = [e * 100 for e in PCK_ESIKLER]
    plt.plot(esik_cm, val_pck, 'o-', color='tab:purple')
    for ec, p in zip(esik_cm, val_pck):
        plt.text(ec, p + 1.5, f'{p:.0f}%', ha='center', fontsize=8)
    plt.xlabel('Eşik (cm)'); plt.ylabel('Doğru Eklem Yüzdesi — PCK (%)')
    plt.title('PCK Eğrisi (Validation E04)')
    plt.ylim(0, 100); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("grafikler/pck_egrisi.png", dpi=150); plt.close()

    print("Grafikler './grafikler/' klasörüne kaydedildi:")
    print("  loss_egrileri.png, val_mpjpe_egrisi.png, eksen_mae_bar.png,")
    print("  model_vs_taban.png, pck_egrisi.png")
    print(f"\nPCK (E04): " +
          ", ".join(f"{c:.1f}cm:%{p:.0f}" for c, p in zip(esik_cm, val_pck)))
