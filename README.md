> У цьому файлі наведено опис дипломного проєкту. За потреби адаптуйте окремі розділи (наприклад, дані керівника).

# 📘 Fractal Robustness – аналіз стійкості ResNet‑18 до шумів на основі фрактальних методів

Проєкт присвячений аналізу стійкості (robustness) згорткової нейромережі ResNet‑18 до різних типів шумів та корупцій на наборах CIFAR‑10 та CIFAR‑10‑C із використанням фрактальних методів:
- оцінка фрактальної розмірності (FD) вхідних зображень, активацій і градієнтів;
- генерація фрактального шуму (2D fBm) як аугментації;
- оцінка робастності за метриками accuracy, CE/mCE, ECE.

---

## 👤 Author / Автор

- **ПІБ**: Козиний Андрій Сергійович  
- **Group / Група**: ФеП‑42  
- **Supervisor / Керівник**: *Свелеба Сергій Андрійович*  
- **Дата виконання**: *14.06.2026*  

---

## 📌 Project Info / Загальна інформація

- **Тип проєкту**: Research / дипломний проєкт (Python ML / Deep Learning)
- **Мова програмування**: Python 3.11
- **Фреймворки / Бібліотеки**:
  - PyTorch, TorchVision
  - NumPy
  - Matplotlib
  - SciPy (опційно)
  - tqdm

---

## 🧠 Functionality Overview / Опис функціоналу

Основні можливості проєкту:

- 🔧 **Тренування моделей на CIFAR‑10**:
  - базова модель ResNet‑18 (Base);
  - модель з фрактальною аугментацією (FBM‑AUG), де до входу додається 2D fBm‑шум.

- 🧪 **Evaluation on CIFAR‑10‑C**:
  - 15 типів корупцій (gaussian_noise, defocus_blur, fog, jpeg_compression, …);
  - 5 рівнів інтенсивності (severity 1–5);
  - обчислення metрик: top‑1 accuracy, CE/mCE, ECE;
  - збереження результатів у CSV для подальшого аналізу.

- 🌪 **Custom noise corruptions (CIFAR‑10)**:
  - власні шуми: Gaussian, speckle (multiplicative), salt‑and‑pepper, defocus blur, JPEG‑like;
  - 5 рівнів інтенсивності для кожного типу шуму;
  - оцінка падіння точності в залежності від рівня шуму.

- 📏 **Fractal analysis / Фрактальний аналіз**:
  - обчислення фрактальної розмірності (FD) методом box‑counting:
    - для вхідних зображень CIFAR‑10;
    - для активацій проміжних шарів ResNet‑18;
    - для градієнт‑карт;
  - аналіз зміни FD уздовж глибини мережі.

- 🖥️ **Console demo / Консольне меню** (`main.py`):
  - меню з опціями: тренування, оцінка на CIFAR‑10‑C, оцінка на custom‑шумах, запуск фрактального аналізу.

---

## 🧱 Project Structure / Основні файли та модулі

| File / Module                         | Description / Призначення                                      |
|--------------------------------------|-----------------------------------------------------------------|
| `fractal_robustness_templates.py`    | Фрактальна розмірність (`fd_boxcount`), fBm (`fbm2d`), CIFAR‑10‑C пайплайн, CIFARResNet18 |
| `fractal_robustness_templates_B.py`  | Тренування на CIFAR‑10 (`TrainConfig`, `train_cifar10`), ECE, метрики CIFAR‑10‑C |
| `fractal_robustness_templates_BB.py` | Розширені метрики CIFAR‑10‑C, baseline‑и для mCE, quickstart‑хелпери |
| `train_baselines.py`                 | Навчання базової та FBM‑AUG моделей ResNet‑18                  |
| `run_cifar10c_eval.py`               | Оцінка моделей на CIFAR‑10‑C, вивід macro‑метрик               |
| `save_results.py`                    | Оцінка CIFAR‑10‑C + custom‑шумів, збереження результатів у CSV |
| `custom_corruptions.py`              | Власні шумові корупції: Gaussian, speckle, salt‑pepper, defocus, JPEG‑like |
| `fractal_analysis.py`                | Зняття активацій/градієнтів і обчислення FD (inputs/acts/grads) |
| `plot_training_curves.py`            | Побудова кривих навчання (loss/accuracy vs epoch)              |
| `plot_cifar10c_gaussian.py`          | Графік accuracy vs severity для корупції `gaussian_noise` (CIFAR‑10‑C) |
| `plot_custom_noises.py`              | Графік accuracy vs severity для custom Gaussian/Speckle/Salt‑Pepper |
| `make_cifar10c_example.py`           | Приклади корупцій CIFAR‑10‑C (колаж із зображень)              |
| `make_custom_noises_example.py`      | Приклади власних шумів (колаж із зображень)                    |
| `main.py`                            | Консольне меню запуску: train / CIFAR‑10‑C eval / custom noise / fractal analysis |

---

## ▶️ How to Run / Як запустити проєкт «з нуля»

### 1. Clone repo / Клонування репозиторію

```bash
git clone <URL_до_репозиторію>
cd FractalRobustness
