"""
=============================================================
 E03 TEST — gerçek vs tahmin görselleştirme + animasyon
=============================================================
 E03 sona saklanan TARAFSIZ test ortamıdır (eğitimde ve validation'da
 kullanılmadı). En iyi model (validation E04'e göre seçilen) yüklenir ve:
   1) RASTGELE birkaç örnekte gerçek (yeşil) vs tahmin (kırmızı) iskeleti
      yan yana ekrana basar + PNG kaydeder
   2) ardışık karelerden ANİMASYONLU video (.mp4, olmazsa .gif) üretir
   3) E03 için MPJPE + eksen-MAE basar (sayısal sonuç)
=============================================================
"""

import copy
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from mmfi import make_dataset
from file3_model import CSITransformerEncoder

# MMFi 17-nokta iskelet bağlantıları
BAGLANTILAR = [
    (0, 1), (1, 2), (2, 3),       # sağ bacak
    (0, 4), (4, 5), (5, 6),       # sol bacak
    (0, 7), (7, 8), (8, 9),       # omurga + baş
    (8, 11), (11, 12), (12, 13),  # sol kol
    (8, 14), (14, 15), (15, 16),  # sağ kol
]
N_RASTGELE = 6      # ekrana basılacak rastgele örnek sayısı
N_ANIM = 120        # animasyon kare sayısı


def hazirla(data, device):
    wifi = data['input_wifi-csi'].to(device).float()
    iskelet = data['output'].to(device).float()
    B = wifi.size(0)
    if wifi.is_complex():
        wifi = torch.view_as_real(wifi)
    wifi = wifi.permute(0, 2, 1, 3, *range(4, wifi.dim()))
    wifi_tokenlari = wifi.reshape(B, 114, -1)
    merkezli = iskelet - iskelet[:, 0:1, :]
    return wifi_tokenlari, merkezli


def iskelet_ciz(ax, poz, renk, etiket, marker='o'):
    ax.scatter(poz[:, 0], poz[:, 1], poz[:, 2], c=renk, marker=marker, s=30, label=etiket)
    for a, b in BAGLANTILAR:
        ax.plot([poz[a, 0], poz[b, 0]],
                [poz[a, 1], poz[b, 1]],
                [poz[a, 2], poz[b, 2]], c=renk, linewidth=2, alpha=0.7)


def eksen_ayarla(ax, baslik):
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y (derinlik)'); ax.set_zlabel('Z')
    ax.set_title(baslik, fontsize=9)


if __name__ == '__main__':
    dataset_root = r"C:\Wifi-Sens data\MMFi_Dataset"

    with open('config.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    # TEST = E03 (config'i bellekte E03'e çevir; diske dokunmaz)
    cfg = copy.deepcopy(config)
    cfg['cross_scene_split']['val_dataset']['scenes'] = ['E03']
    _, test_dataset = make_dataset(dataset_root, cfg)
    print(f"Test (E03): {len(test_dataset)} örnek")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    model = CSITransformerEncoder(raw_token_dim=30).to(device)
    # en iyi model yoksa son modele düş
    # NOT: 'best' model 1. epoch'a denk geldi (validation MPJPE eğrisi 1. epoch'ta
    # en düşüktü ama o model neredeyse eğitilmemişti). O yüzden gerçekten eğitilmiş
    # 30. epoch modelini ('son') kullanıyoruz; testin anlamlı olması için bu doğru.
    model.load_state_dict(torch.load("wifi_iskelet_modeli_son.pth",
                                     map_location=device, weights_only=True))
    print("Model yüklendi: wifi_iskelet_modeli_son.pth (30. epoch — tam eğitilmiş)")
    model.eval()

    # --- Sayısal sonuç (E03) ---
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=True)
    with torch.no_grad():
        mpjpe_sum = 0.0
        eksen_sum = torch.zeros(3, device=device)
        tj = 0
        for i, data in enumerate(test_loader):
            if i >= 400:
                break
            wifi_tokenlari, merkezli = hazirla(data, device)
            B = wifi_tokenlari.size(0)
            _, pred_kp, _ = model(wifi_tokenlari)
            pred_kp = pred_kp.view(B, 17, 3)
            mpjpe_sum += torch.norm(pred_kp - merkezli, dim=-1).sum().item()
            eksen_sum += (pred_kp - merkezli).abs().sum(dim=(0, 1))
            tj += B * 17
    mpjpe = mpjpe_sum / tj * 1000
    eksen = (eksen_sum / tj * 1000).cpu().tolist()
    print("\n" + "=" * 56)
    print("*** E03 TEST SONUÇLARI (tarafsız) ***")
    print(f"  MPJPE: {mpjpe:.1f} mm | X/Y/Z MAE: "
          f"{eksen[0]:.1f} / {eksen[1]:.1f} / {eksen[2]:.1f} mm")
    print("=" * 56 + "\n")

    # --- Görselleştirme için karıştırılmış bir loader (rastgele örnek) ---
    vis_loader = DataLoader(test_dataset, batch_size=max(N_RASTGELE, N_ANIM),
                            shuffle=True, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for data in vis_loader:
            wifi_tokenlari, merkezli = hazirla(data, device)
            B = wifi_tokenlari.size(0)
            _, pred_kp, _ = model(wifi_tokenlari)
            tahmin = pred_kp.view(B, 17, 3)
            gercek = merkezli
            break
    gercek = gercek.cpu().numpy()
    tahmin = tahmin.cpu().numpy()

    # 1) RASTGELE ÖRNEKLER — gerçek vs tahmin yan yana
    print(f"{N_RASTGELE} rastgele örnek çiziliyor...")
    sutun = 3
    satir = int(np.ceil(N_RASTGELE / sutun))
    fig = plt.figure(figsize=(5 * sutun, 4.5 * satir))
    for idx in range(N_RASTGELE):
        ax = fig.add_subplot(satir, sutun, idx + 1, projection='3d')
        iskelet_ciz(ax, gercek[idx], 'green', 'Gerçek (GT)', marker='o')
        iskelet_ciz(ax, tahmin[idx], 'red', 'Tahmin (WiFi)', marker='x')
        mp = np.linalg.norm(tahmin[idx] - gercek[idx], axis=-1).mean() * 1000
        eksen_ayarla(ax, f"Örnek {idx+1} — MPJPE {mp:.0f} mm")
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    plt.suptitle("E03 Test — Gerçek (yeşil) vs Tahmin (kırmızı)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("e03_rastgele_ornekler.png", dpi=150)
    print("  -> e03_rastgele_ornekler.png kaydedildi")
    plt.show()

    # 2) ANİMASYON — ardışık kareler
    print(f"Animasyon ({N_ANIM} kare) hazırlanıyor...")
    n_anim = min(N_ANIM, gercek.shape[0])
    fig_anim = plt.figure(figsize=(8, 7))
    ax_anim = fig_anim.add_subplot(111, projection='3d')

    def update(frame):
        ax_anim.clear()
        iskelet_ciz(ax_anim, gercek[frame], 'green', 'Gerçek', 'o')
        iskelet_ciz(ax_anim, tahmin[frame], 'red', 'Tahmin', 'x')
        mp = np.linalg.norm(tahmin[frame] - gercek[frame], axis=-1).mean() * 1000
        eksen_ayarla(ax_anim, f"E03 Test — Kare {frame+1}/{n_anim} — MPJPE {mp:.0f} mm")
        ax_anim.legend(fontsize=8, loc='upper right')

    ani = FuncAnimation(fig_anim, update, frames=range(n_anim), interval=100)

    # önce mp4 (ffmpeg), olmazsa gif (pillow)
    kaydedildi = False
    try:
        from matplotlib.animation import FFMpegWriter
        ani.save("e03_animasyon.mp4", writer=FFMpegWriter(fps=10, bitrate=1800))
        print("  -> e03_animasyon.mp4 kaydedildi")
        kaydedildi = True
    except Exception as e:
        print(f"  mp4 kaydedilemedi ({type(e).__name__}); gif deneniyor...")
    if not kaydedildi:
        try:
            from matplotlib.animation import PillowWriter
            ani.save("e03_animasyon.gif", writer=PillowWriter(fps=10))
            print("  -> e03_animasyon.gif kaydedildi")
        except Exception as e:
            print(f"  gif de kaydedilemedi ({type(e).__name__}). "
                  f"pip install pillow / ffmpeg kurmayı dene.")
    plt.show()
    print("\nBitti.")
