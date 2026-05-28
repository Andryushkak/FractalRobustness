> У цьому файлі наведено опис дипломного проєкту. За потреби адаптуйте окремі розділи (наприклад, дані керівника, рік, шлях до репозиторію).

# 📘 Fractal Robustness – аналіз стійкості ResNet‑18 до шумів на основі фрактальних методів

Проєкт присвячений аналізу стійкості (robustness) згорткової нейромережі **ResNet‑18** до різних типів шумів та корупцій на наборах **CIFAR‑10** та **CIFAR‑10‑C** із використанням фрактальних методів:
- оцінка фрактальної розмірності (FD) вхідних зображень, активацій і градієнтів;
- генерація фрактального шуму (2D fBm) як аугментації;
- оцінка робастності за метриками **accuracy**, **CE/mCE**, **ECE**.

---

## 👤 Author / Автор

- **ПІБ:** **Козиний Андрій Сергійович**  
- **Group / Група:** **ФеП‑42**  
- **Supervisor / Керівник:** **Свелеба Сергій Андрійович**  
- **Дата виконання:** **14.05.2026**  

---

## 📌 Project Info / Загальна інформація

- **Тип проєкту:** дипломний дослідницький (Python / Deep Learning)
- **Мова програмування:** **Python 3.11**
- **Фреймворки / Бібліотеки:**
  - **PyTorch**, **TorchVision**
  - **NumPy**
  - **Matplotlib**
  - **SciPy** (опційно)
  - **tqdm**

---

## 🧠 Functionality Overview / Опис функціоналу

Основні можливості проєкту:

- 🔧 **Training on CIFAR‑10 / Тренування на CIFAR‑10**
  - **Base**: ResNet‑18 зі стандартними аугментаціями (RandomCrop, RandomHorizontalFlip, нормалізація);
  - **FBM‑AUG**: ті ж аугментації + 2D fBm‑noise через `AddFBmNoise(H=0.6, sigma=0.03)`.

- 🧪 **Evaluation on CIFAR‑10‑C**
  - 15 типів корупцій (gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression);
  - 5 рівнів інтенсивності (severity 1–5);
  - метрики: top‑1 **accuracy**, **CE/mCE**, **ECE**;
  - збереження результатів у CSV (`results_cifar10c_base.csv`, `results_cifar10c_fbm.csv`).

- 🌪 **Custom noise corruptions (CIFAR‑10)**
  - власні шуми: **Gaussian**, **speckle** (multiplicative), **salt‑and‑pepper**, **defocus blur**, **JPEG‑like**;
  - 5 рівнів інтенсивності для кожного типу шуму (s1..s5);
  - результати в `results_custom_noises_base.csv` + графіки `accuracy vs severity`.

- 📏 **Fractal analysis / Фрактальний аналіз**
  - обчислення **фрактальної розмірності (FD)** методом box‑counting:
    - для вхідних зображень CIFAR‑10;
    - для активацій вибраних шарів ResNet‑18;
    - для градієнт‑карт;
  - аналіз зміни FD уздовж глибини мережі.

- 🖥️ **Console demo / Консольне меню** (`main.py`)
  - інтерактивне меню з опціями:
    1. Train baselines (clean + fBm)
    2. Evaluate on CIFAR‑10‑C (base + fBm)
    3. Evaluate on custom noises
    4. Fractal analysis

---

## 🧱 Project Structure / Основні файли та модулі

| File / Module                         | Description / Призначення                                      |
|--------------------------------------|-----------------------------------------------------------------|
| `fractal_robustness_templates.py`    | Фрактальна розмірність (`fd_boxcount`), fBm (`fbm2d`), CIFAR‑10‑C пайплайн, `CIFARResNet18` |
| `fractal_robustness_templates_B.py`  | Тренування CIFAR‑10 (`TrainConfig`, `train_cifar10`), ECE, CIFAR‑10‑C metrics |
| `fractal_robustness_templates_BB.py` | Розширені метрики CIFAR‑10‑C, baseline‑и для mCE, quickstart‑хелпери |
| `train_baselines.py`                 | Навчання базової та FBM‑AUG моделей ResNet‑18                  |
| `run_cifar10c_eval.py`               | Оцінка моделей на CIFAR‑10‑C, вивід macro‑метрик у консоль     |
| `save_results.py`                    | CIFAR‑10‑C + custom‑шуми → збереження результатів у CSV        |
| `custom_corruptions.py`              | Власні шуми: Gaussian, speckle, salt_pepper, defocus, JPEG‑like|
| `fractal_analysis.py`                | Зняття активацій/градієнтів і обчислення FD (inputs/acts/grads)|
| `plot_training_curves.py`            | Побудова кривих навчання (loss / accuracy vs epoch)            |
| `plot_cifar10c_gaussian.py`          | Графік `accuracy vs severity` для `gaussian_noise` (Base vs FBM‑AUG) |
| `plot_custom_noises.py`              | Графік `accuracy vs severity` для custom Gaussian/Speckle/Salt‑Pepper |
| `make_cifar10c_example.py`           | Колаж CIFAR‑10‑C корупцій                                      |
| `make_custom_noises_example.py`      | Колаж власних шумів                                            |
| `main.py`                            | Консольне меню (train / CIFAR‑10‑C eval / custom noise / FD)   |

---

## ▶️ How to Run / Як запустити проєкт «з нуля»

### 1. Clone repo / Клонування репозиторію

```bash
git clone https://github.com/Andryushkak/FractalRobustness.git
cd FractalRobustness
