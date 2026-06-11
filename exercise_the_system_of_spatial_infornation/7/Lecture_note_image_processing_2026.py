import os
import glob
import tarfile
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# Data download / extraction
# ================================================================
TRAIN_PATH = './lecture_data/'
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)
if not os.path.exists(TRAIN_PATH):
    if not os.path.exists('lecture_data.tar'):
        print("lecture_data.tar をダウンロード中...")
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
        import gdown
        gdown.download(id='19_IbdV3xtpqZYifyMVrcQE4liS8zVp6t', output='lecture_data.tar', quiet=False)
    print("lecture_data.tar を展開中...")
    with tarfile.open('lecture_data.tar') as tar:
        tar.extractall('.')
    print("展開完了")

# ================================================================
# Common imports
# ================================================================
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

folder = ["000_damage", "001_no_damage"]
CLASS_NUM = len(folder)

print('The number of damage images: {}'.format(len(os.listdir(os.path.join(TRAIN_PATH, folder[0])))))
print('The number of no damage images: {}'.format(len(os.listdir(os.path.join(TRAIN_PATH, folder[1])))))

# ================================================================
# Image manipulation demos
# ================================================================
sample_file = os.path.join(TRAIN_PATH, folder[0],
    'SECTION_1_Promotion_0_500 _Output_Range_1_chunksize_256_1_1_crack_easy.png')
img_demo = img_to_array(load_img(sample_file, color_mode="rgb", target_size=(256, 256))) / 255.0

print(img_demo)
print(img_demo.shape)

def horizontal_flip(image):
    return image[:, ::-1, :]

def vertical_flip(image, rate=1):
    if np.random.rand() < rate:
        image = image[::-1, :, :]
    return image

def random_crop(image, crop_size=(128, 128)):
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    return image[top:top + crop_size[0], left:left + crop_size[1], :]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img_demo);                    axes[0].set_title('Original')
axes[1].imshow(horizontal_flip(img_demo));   axes[1].set_title('H-Flip')
axes[2].imshow(vertical_flip(img_demo));     axes[2].set_title('V-Flip')
axes[3].imshow(random_crop(img_demo));       axes[3].set_title('Random Crop')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'image_augmentation.png'))
plt.close()
print("Saved: image_augmentation.png")

# ================================================================
# Shared helpers
# ================================================================
def load_dataset(color_channel, input_image_size):
    v_image, v_label = [], []
    for index, name in enumerate(folder):
        d = os.path.join(TRAIN_PATH, name)
        files = glob.glob(os.path.join(d, "*.png"))
        print(d)
        for file in files:
            mode = "grayscale" if color_channel == 1 else "rgb"
            img = load_img(file, color_mode=mode, target_size=(input_image_size, input_image_size))
            v_image.append(img_to_array(img))
            v_label.append(index)
    v_image = np.array(v_image, dtype='float32') / 255.0
    v_label = to_categorical(np.array(v_label), CLASS_NUM)
    return v_image, v_label

def split_dataset(v_image, v_label, random_seed=1):
    _train, test, _train_l, test_l = train_test_split(
        v_image, v_label, test_size=0.20, random_state=random_seed)
    train, val, train_l, val_l = train_test_split(
        _train, _train_l, test_size=0.25, random_state=random_seed)
    return train, val, test, train_l, val_l, test_l

def plot_history(history, title, filename):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    fig.suptitle(title)
    axL.plot(history.history['loss'],     label="train loss")
    axL.plot(history.history['val_loss'], label="val loss")
    axL.set_title('Loss'); axL.set_xlabel('epoch'); axL.legend()
    axR.plot(history.history['accuracy'],     label="train acc")
    axR.plot(history.history['val_accuracy'], label="val acc")
    axR.set_title('Accuracy'); axR.set_xlabel('epoch'); axR.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

def evaluate_and_report(model, model_inputs, test_labels, display_images, model_name):
    os.makedirs(os.path.join(RESULTS_DIR, model_name), exist_ok=True)
    score = model.evaluate(model_inputs, test_labels, verbose=0)
    preds_raw = model.predict(model_inputs)
    preds = [0 if p[0] > 0.5 else 1 for p in preds_raw]
    true = np.argmax(test_labels, axis=1)

    cm = confusion_matrix(true, preds)
    cr = classification_report(true, preds, target_names=folder, output_dict=True)
    cr_text = classification_report(true, preds, target_names=folder)

    print(f"\n{'='*50}")
    print(f"[{model_name}] Evaluation")
    print(f"{'='*50}")
    print(f"Test Loss: {score[0]:.4f}   Test Accuracy: {score[1]:.4f}")
    print("Confusion Matrix")
    print(cm)
    print("Classification Report")
    print(cr_text)

    fn_counter = fp_counter = tp_counter = tn_counter = 0
    for index, value in enumerate(true):
        tag = None
        if value == 0 and preds[index] == 0: tag = f'TP_{index}'; tp_counter += 1
        if value == 1 and preds[index] == 1: tag = f'TN_{index}'; tn_counter += 1
        if value == 0 and preds[index] == 1: tag = f'FN_{index}'; fn_counter += 1
        if value == 1 and preds[index] == 0: tag = f'FP_{index}'; fp_counter += 1
        if tag:
            disp = display_images[index]
            if disp.ndim == 1:
                side = int(disp.shape[0] ** 0.5)
                disp = disp.reshape(side, side)
            _, ax = plt.subplots(figsize=(4, 4))
            ax.set_title(f'{model_name} {tag}')
            ax.imshow(disp, cmap='gray' if disp.ndim == 2 else None)
            plt.savefig(os.path.join(RESULTS_DIR, model_name, f'error_{tag}.png'))
            plt.close()

    print(f"TP: {tp_counter}  TN: {tn_counter}  FN: {fn_counter}  FP: {fp_counter}")
    return {
        'loss': score[0],
        'accuracy': score[1],
        'confusion_matrix': cm,
        'classification_report_dict': cr,
        'classification_report_text': cr_text,
        'tp': tp_counter, 'tn': tn_counter,
        'fn': fn_counter, 'fp': fp_counter,
    }

results = {}

# ================================================================
# Model 1: Simple Neural Network (grayscale, 256x256, flatten)
# ================================================================
print("\n" + "="*50)
print("Model 1: Simple Neural Network")
print("="*50)

COLOR_CHANNEL_NN = 1
INPUT_IMAGE_SIZE_NN = 256

v_image, v_label = load_dataset(COLOR_CHANNEL_NN, INPUT_IMAGE_SIZE_NN)
train_img, val_img, test_img, train_l, val_l, test_l = split_dataset(v_image, v_label)

flat = INPUT_IMAGE_SIZE_NN * INPUT_IMAGE_SIZE_NN
train_flat = train_img.reshape(train_img.shape[0], flat)
val_flat   = val_img.reshape(val_img.shape[0], flat)
test_flat  = test_img.reshape(test_img.shape[0], flat)

print(train_flat.shape, val_flat.shape, test_flat.shape)

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(flat,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax'),
])
nn_model.summary()
#model.compile(
#    loss='categorical_crossentropy',
#    optimizer=Adam(lr=0.00001),
#    metrics=['accuracy'])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.00001),
                 loss='categorical_crossentropy', metrics=['accuracy'])

fit_nn = nn_model.fit(train_flat, train_l, batch_size=128, epochs=200,
                      verbose=2, validation_data=(val_flat, val_l))

plot_history(fit_nn, 'Simple NN', 'history_simple_nn.png')
nn_result = evaluate_and_report(
    nn_model, test_flat, test_l,
    display_images=test_img.reshape(test_img.shape[0], INPUT_IMAGE_SIZE_NN, INPUT_IMAGE_SIZE_NN),
    model_name='SimpleNN')
results['Simple NN'] = nn_result

# ================================================================
# Model 2: CNN (RGB, 256x256)
# ================================================================
print("\n" + "="*50)
print("Model 2: CNN")
print("="*50)

COLOR_CHANNEL_CNN = 3
INPUT_IMAGE_SIZE_CNN = 256
BATCH_SIZE_CNN = 8
EPOCH_NUM_CNN = 5
print(f"BATCH_SIZE: {BATCH_SIZE_CNN}  EPOCH_NUM: {EPOCH_NUM_CNN}")

v_image, v_label = load_dataset(COLOR_CHANNEL_CNN, INPUT_IMAGE_SIZE_CNN)
train_img, val_img, test_img, train_l, val_l, test_l = split_dataset(v_image, v_label)

def create_cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(
        input_shape=(INPUT_IMAGE_SIZE_CNN, INPUT_IMAGE_SIZE_CNN, COLOR_CHANNEL_CNN)))
    model.add(tf.keras.layers.Conv2D(64, (7, 7), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Activation('elu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Activation('softmax'))
    return model

gen_cnn = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                              width_shift_range=0.1, height_shift_range=0.1,
                              zoom_range=0.1, rotation_range=10)
generator_cnn = gen_cnn.flow(train_img, train_l, batch_size=BATCH_SIZE_CNN)

cnn_model = create_cnn_model()
cnn_model.summary()
cnn_model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
                  metrics=['accuracy'])

history_cnn = cnn_model.fit(generator_cnn, epochs=EPOCH_NUM_CNN,
                             steps_per_epoch=train_img.shape[0] // BATCH_SIZE_CNN,
                             validation_data=(val_img, val_l))

plot_history(history_cnn, 'CNN', 'history_cnn.png')
cnn_result = evaluate_and_report(
    cnn_model, test_img, test_l,
    display_images=test_img,
    model_name='CNN')
results['CNN'] = cnn_result

# ================================================================
# Model 3: ResNet50 (RGB, 224x224)
# ================================================================
print("\n" + "="*50)
print("Model 3: ResNet50")
print("="*50)

COLOR_CHANNEL_RN = 3
INPUT_IMAGE_SIZE_RN = 224
BATCH_SIZE_RN = 8
EPOCH_NUM_RN = 5
print(f"BATCH_SIZE: {BATCH_SIZE_RN}  EPOCH_NUM: {EPOCH_NUM_RN}")

v_image, v_label = load_dataset(COLOR_CHANNEL_RN, INPUT_IMAGE_SIZE_RN)
# ResNet50 は [0,1] 正規化ではなく専用の前処理が必要
v_image_rn = tf.keras.applications.resnet50.preprocess_input(v_image * 255.0)
train_img, val_img, test_img, train_l, val_l, test_l = split_dataset(v_image_rn, v_label)

def create_resnet_model():
    base_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights='imagenet',
        input_shape=(INPUT_IMAGE_SIZE_RN, INPUT_IMAGE_SIZE_RN, COLOR_CHANNEL_RN)
    )
    base_model.trainable = False  # 事前学習済み重みを固定
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Activation('elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2),
        tf.keras.layers.Activation('softmax'),
    ])
    return model

gen_rn = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                             width_shift_range=0.1, height_shift_range=0.1,
                             zoom_range=0.1, rotation_range=10)
generator_rn = gen_rn.flow(train_img, train_l, batch_size=BATCH_SIZE_RN)

resnet_model = create_resnet_model()
resnet_model.summary()
resnet_model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
                     metrics=['accuracy'])

# opt = tf.keras.optimizers.legacy.RMSprop(
#     learning_rate=0.0001, rho=0.9, momentum=0.0,
#     epsilon=1e-07, centered=False, name='RMSprop', decay=1e-6)

history_rn = resnet_model.fit(generator_rn, epochs=EPOCH_NUM_RN,
                               steps_per_epoch=train_img.shape[0] // BATCH_SIZE_RN,
                               validation_data=(val_img, val_l))
# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/Model

plot_history(history_rn, 'ResNet50', 'history_resnet.png')
# display_images は前処理前の元画像を使う（imshow 用）
_, _, test_img_orig, _, _, _ = split_dataset(v_image, v_label)
rn_result = evaluate_and_report(
    resnet_model, test_img, test_l,
    display_images=test_img_orig,
    model_name='ResNet50')
results['ResNet50'] = rn_result

# ================================================================
# Final comparison
# ================================================================
print("\n" + "="*50)
print("Final Comparison")
print("="*50)
print(f"{'Model':<20} {'Test Loss':>10} {'Test Accuracy':>15}")
print("-" * 47)
for name, r in results.items():
    print(f"{name:<20} {r['loss']:>10.4f} {r['accuracy']:>15.4f}")

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
names  = list(results.keys())
losses = [results[n]['loss']     for n in names]
accs   = [results[n]['accuracy'] for n in names]
axL.bar(names, losses); axL.set_title('Test Loss');     axL.set_ylabel('Loss')
axR.bar(names, accs);   axR.set_title('Test Accuracy'); axR.set_ylabel('Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comparison.png'))
plt.close()
print(f"Saved: {os.path.join(RESULTS_DIR, 'comparison.png')}")

# ================================================================
# Report generation
# ================================================================
import datetime

def _cm_to_md(cm):
    header = "| | Pred: damage | Pred: no_damage |\n|---|---|---|\n"
    row0 = f"| **Actual: damage**    | {cm[0][0]} (TP) | {cm[0][1]} (FN) |\n"
    row1 = f"| **Actual: no_damage** | {cm[1][0]} (FP) | {cm[1][1]} (TN) |\n"
    return header + row0 + row1

def _cr_to_md(cr_dict):
    lines = "| Class | Precision | Recall | F1-score | Support |\n"
    lines += "|---|---|---|---|---|\n"
    for cls in folder:
        d = cr_dict[cls]
        lines += f"| {cls} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1-score']:.3f} | {int(d['support'])} |\n"
    d = cr_dict['weighted avg']
    lines += f"| **weighted avg** | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1-score']:.3f} | {int(d['support'])} |\n"
    return lines

def _best(key):
    return max(results, key=lambda n: results[n][key])

def _worst(key):
    return min(results, key=lambda n: results[n][key])

def generate_report():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_acc_model  = _best('accuracy')
    worst_acc_model = _worst('accuracy')
    best_loss_model = _worst('loss')

    lines = []
    lines.append(f"# Road Damage Classification — Experiment Report")
    lines.append(f"\n**Generated:** {now}\n")

    # ---- Dataset ----
    lines.append("---")
    lines.append("## 1. Dataset")
    n_damage    = len(os.listdir(os.path.join(TRAIN_PATH, folder[0])))
    n_nodamage  = len(os.listdir(os.path.join(TRAIN_PATH, folder[1])))
    total       = n_damage + n_nodamage
    lines.append(f"""
| Class | Count | Ratio |
|---|---|---|
| 000_damage    | {n_damage}   | {n_damage/total*100:.1f}% |
| 001_no_damage | {n_nodamage} | {n_nodamage/total*100:.1f}% |
| **Total**     | **{total}**  | 100% |

- 画像サイズ: 256×256 (RGB) または 224×224 (ResNet)
- Split: Train 60% / Validation 20% / Test 20% (random_seed=1)
- **クラス不均衡に注意:** no_damage が damage より約 {n_nodamage/n_damage:.1f} 倍多い。
  Accuracy だけで評価すると常に多数クラスを予測するだけで高スコアが出る可能性がある。
""")

    # ---- Image augmentation ----
    lines.append("---")
    lines.append("## 2. Image Augmentation")
    lines.append("""
学習データに以下のオーグメンテーションを適用して過学習を抑制した。

| 手法 | 効果 |
|---|---|
| Horizontal Flip | 左右反転による位置不変性の学習 |
| Vertical Flip   | 上下反転（道路画像では稀だが汎化に貢献） |
| Width/Height Shift (±10%) | 小さな位置ずれへの頑健性 |
| Zoom (±10%)     | スケール変動への対応 |
| Rotation (±10°) | 微小な傾きへの頑健性 |

![augmentation](image_augmentation.png)
""")

    # ---- Models ----
    lines.append("---")
    lines.append("## 3. モデル構成")
    lines.append("""
### Model 1: Simple Neural Network
- 入力: 256×256 グレースケール → Flatten (65536次元)
- Dense(512, ReLU) → Dropout(0.2) → Dense(512, ReLU) → Dropout(0.2) → Dense(2, Softmax)
- Optimizer: Adam(lr=0.01) / Epochs: 200 / Batch: 128
- **特徴:** 最もシンプルな全結合ネット。空間情報は完全に失われる。

### Model 2: CNN
- 入力: 256×256 RGB
- BatchNorm → Conv2D(64, 7×7) → MaxPool → Dropout(0.4)
- BatchNorm → Conv2D(128, 3×3) → MaxPool → Dropout(0.4)
- BatchNorm → Conv2D(256, 5×5) → MaxPool → Dropout(0.4)
- Flatten → Dense(256, ELU) → Dropout(0.5) → Dense(2, Softmax)
- Optimizer: AdamW(lr=0.0001) / Epochs: 5 / Batch: 8
- **特徴:** 空間的な特徴（エッジ・テクスチャ）を階層的に抽出する。

### Model 3: ResNet50 (転移学習)
- 入力: 224×224 RGB
- ResNet50 (ImageNet事前学習済み, include_top=True)
- Flatten → Dense(256, ELU) → Dropout(0.5) → Dense(2, Softmax)
- Optimizer: AdamW(lr=0.0001) / Epochs: 5 / Batch: 8
- **特徴:** 2500万パラメータ超の深層ネット。ImageNetの大規模事前学習により、少ないデータでも豊富な特徴表現を利用できる（転移学習）。
""")

    # ---- Per-model results ----
    lines.append("---")
    lines.append("## 4. 各モデルの結果")

    model_meta = {
        'Simple NN': ('SimpleNN',  'history_simple_nn.png', 200, 128, 'Adam(lr=0.01)'),
        'CNN':       ('CNN',       'history_cnn.png',        5,   8,  'AdamW(lr=0.0001)'),
        'ResNet50':  ('ResNet50',  'history_resnet.png',     5,   8,  'AdamW(lr=0.0001)'),
    }

    for i, (name, r) in enumerate(results.items(), 1):
        dir_name, hist_img, epochs, batch, opt = model_meta[name]
        cm = r['confusion_matrix']
        total_test = r['tp'] + r['tn'] + r['fn'] + r['fp']
        lines.append(f"\n### 4-{i}. {name}\n")
        lines.append(f"**Hyperparameters:** Epochs={epochs}, Batch={batch}, Optimizer={opt}\n")
        lines.append(f"#### 学習曲線\n![history]({hist_img})")
        lines.append(f"""
#### テスト結果

| Metric | Value |
|---|---|
| Test Loss     | {r['loss']:.4f} |
| Test Accuracy | {r['accuracy']:.4f} ({r['accuracy']*100:.1f}%) |
| TP (正解: damage,    予測: damage)    | {r['tp']} |
| TN (正解: no_damage, 予測: no_damage) | {r['tn']} |
| FN (damage を見逃し)                  | {r['fn']} |
| FP (no_damage を damage と誤検知)     | {r['fp']} |
| テスト画像数                          | {total_test} |
""")
        lines.append("#### Confusion Matrix\n")
        lines.append(_cm_to_md(cm))
        lines.append("\n#### Classification Report\n")
        lines.append(_cr_to_md(r['classification_report_dict']))

        # コメント
        prec_dmg = r['classification_report_dict'][folder[0]]['precision']
        rec_dmg  = r['classification_report_dict'][folder[0]]['recall']
        f1_dmg   = r['classification_report_dict'][folder[0]]['f1-score']
        lines.append(f"""
#### コメント
- **Accuracy {r['accuracy']*100:.1f}%** は一見{('高い' if r['accuracy']>0.7 else '低い')}が、クラス不均衡（no_damage多数）を考慮する必要がある。
- damage クラスの **Recall = {rec_dmg:.3f}** ({rec_dmg*100:.1f}%) ← 実際の損傷のうち何割を検出できたか。道路管理の観点では**最重要指標**。見逃し(FN)={r['fn']}件。
- damage クラスの **Precision = {prec_dmg:.3f}** ← damage と予測したうち本当に損傷だった割合。誤警報(FP)={r['fp']}件。
- **F1-score (damage) = {f1_dmg:.3f}** ← PrecisionとRecallの調和平均。クラス不均衡下での総合指標。
""")

    # ---- Comparison ----
    lines.append("---")
    lines.append("## 5. モデル比較\n")
    lines.append("![comparison](comparison.png)\n")
    lines.append(f"| Model | Test Loss | Test Accuracy | damage F1 | damage Recall | FN (見逃し) | FP (誤検知) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        f1  = r['classification_report_dict'][folder[0]]['f1-score']
        rec = r['classification_report_dict'][folder[0]]['recall']
        lines.append(f"| {name} | {r['loss']:.4f} | {r['accuracy']:.4f} | {f1:.3f} | {rec:.3f} | {r['fn']} | {r['fp']} |")

    lines.append(f"""
### 考察

- **Accuracy 最高:** {best_acc_model}
- **Loss 最小:** {best_loss_model}
- **damage Recall 最高:** {max(results, key=lambda n: results[n]['classification_report_dict'][folder[0]]['recall'])}

道路損傷検出では**見逃し(FN)のコストが最も高い**（損傷を放置すると事故リスク）。
そのため Accuracy よりも **damage クラスの Recall** を重視して最終モデルを選定すべきである。

転移学習(ResNet50)はパラメータ数が多い一方、今回のような小規模データセット({total}枚)では
少ないエポック数でも十分な特徴を引き出せる可能性がある。
一方 Simple NN は空間情報を失うため、画像分類には構造的に不利である。
CNNは今回のデータ規模に適した中間的な選択肢といえる。
""")

    lines.append("---")
    lines.append("## 6. 出力ファイル一覧\n")
    lines.append("""
| ファイル | 内容 |
|---|---|
| `image_augmentation.png` | オーグメンテーション例（Original/H-Flip/V-Flip/Crop） |
| `history_simple_nn.png` | Simple NN の学習曲線 |
| `history_cnn.png` | CNN の学習曲線 |
| `history_resnet.png` | ResNet50 の学習曲線 |
| `comparison.png` | 3モデルのLoss・Accuracy棒グラフ比較 |
| `SimpleNN/error_TP_*.png` | Simple NNの正解(TP)画像 |
| `SimpleNN/error_TN_*.png` | Simple NNの正解(TN)画像 |
| `SimpleNN/error_FN_*.png` | Simple NNの見逃し(FN)画像 |
| `SimpleNN/error_FP_*.png` | Simple NNの誤検知(FP)画像 |
| `CNN/error_*.png` | CNN の誤分類・正解画像 |
| `ResNet50/error_*.png` | ResNet50 の誤分類・正解画像 |
| `report.md` | 本レポート |
""")

    report_path = os.path.join(RESULTS_DIR, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved: {report_path}")

generate_report()
