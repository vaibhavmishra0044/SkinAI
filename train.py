"""
SkinAI -- HAM10000 Feature-Extraction Training (CPU-optimised)
==============================================================

WHY THIS APPROACH (not end-to-end training):
  End-to-end EfficientNetV2B0 at 224x224 on CPU = ~2 hours/epoch.
  Feature-extraction approach:
    1. Run frozen backbone over all images ONCE (~10 min)  -> 1280-dim vectors
    2. Train dense head on those vectors  -> ~5 min total for 200 epochs
    3. Assemble backbone+head -> save .keras
    4. Optional Stage-2: fine-tune last 30 backbone layers

ROOT CAUSES OF ORIGINAL LOW-CONFIDENCE / WRONG PREDICTIONS:
  1. No oversampling  -> model collapsed onto melanocytic_nevi (67% of data)
  2. No class weights -> df=1.1%, vasc=1.4% never learned
  3. No backbone fine-tuning (or with wrong LR)
  4. bare keras imports -> MemoryError
  5. Preprocessing mismatch: PIL LANCZOS vs TF bilinear

ALL FIXED HERE:
  - tensorflow.keras only
  - Aggressive oversampling: every class >= MIN_PER_CLASS samples
  - Class weights on loss
  - Random augmentation per oversampled copy (unique feature vectors)
  - Label smoothing 0.1
  - Same preprocessing as app.py (tf.image.resize bilinear + preprocess_input)
  - Stage-2 fine-tune for extra accuracy
  - Per-class accuracy saved to model_metadata.json
"""

import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback,
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset can be: HAM10000_DIR env var, inside project/archive, or ../archive (sibling folder)
_env_dir = os.getenv("HAM10000_DIR")
if _env_dir:
    DATA_DIR = _env_dir
elif os.path.isdir(os.path.join(PROJECT_ROOT, "archive")):
    DATA_DIR = os.path.join(PROJECT_ROOT, "archive")
else:
    DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "archive")
META_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
IMAGE_DIRS = [
    os.path.join(DATA_DIR, "images"),
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]

IMAGE_SIZE    = (224, 224)
FEAT_BATCH    = int(os.getenv("HAM_FEAT_BATCH",     "64"))
HEAD_BATCH    = int(os.getenv("HAM_HEAD_BATCH",     "256"))
HEAD_EPOCHS   = int(os.getenv("HAM_HEAD_EPOCHS",    "200"))
FT_EPOCHS     = int(os.getenv("HAM_FT_EPOCHS",      "12"))
FT_LR         = float(os.getenv("HAM_FT_LR",        "3e-6"))
MIN_PER_CLASS = int(os.getenv("HAM_MIN_OVERSAMPLE",  "1500"))
LABEL_SMOOTH  = 0.10
SEED          = 42

CLASS_NAMES = [
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis_like_lesions",
    "dermatofibroma",
    "melanoma",
    "melanocytic_nevi",
    "vascular_lesions",
]
NUM_CLASSES = len(CLASS_NAMES)

DX_MAP = {"akiec":0,"bcc":1,"bkl":2,"df":3,"mel":4,"nv":5,"vasc":6}


def list_images():
    img_map = {}
    for base in IMAGE_DIRS:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for name in files:
                if name.lower().endswith((".jpg",".jpeg",".png")):
                    img_map[os.path.splitext(name)[0]] = os.path.join(root, name)
    return img_map


def load_records(img_map):
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"HAM10000_metadata.csv not found at {META_PATH}")
    records = []
    with open(META_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            img_id = row.get("image_id","")
            dx     = row.get("dx","")
            if dx in DX_MAP and img_id in img_map:
                records.append((img_map[img_id], DX_MAP[dx]))
    if not records:
        raise ValueError("No records matched. Check dataset paths.")
    return records


def stratified_split(records, train_r=0.80, val_r=0.10):
    grp = defaultdict(list)
    for path, lbl in records:
        grp[lbl].append(path)
    random.seed(SEED)
    train, val, test = [], [], []
    for lbl, paths in grp.items():
        random.shuffle(paths)
        n = len(paths)
        t1, t2 = int(n*train_r), int(n*(train_r+val_r))
        train += [(p,lbl) for p in paths[:t1]]
        val   += [(p,lbl) for p in paths[t1:t2]]
        test  += [(p,lbl) for p in paths[t2:]]
    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return train, val, test


def oversample(records, min_per_class=MIN_PER_CLASS):
    grp = defaultdict(list)
    for path, lbl in records:
        grp[lbl].append(path)
    max_n  = max(len(v) for v in grp.values())
    target = max(min_per_class, max_n // 2)
    balanced = []
    for lbl in range(NUM_CLASSES):
        items = grp.get(lbl, [])
        if not items:
            continue
        if len(items) >= target:
            balanced.extend([(p,lbl) for p in items])
        else:
            reps = (target // len(items)) + 1
            copies = (items * reps)[:target]
            balanced.extend([(p,lbl) for p in copies])
    random.shuffle(balanced)
    return balanced


def class_weights_from_records(records):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, lbl in records:
        counts[lbl] += 1
    total = counts.sum()
    return {int(i): float(total)/(NUM_CLASSES*max(c,1)) for i,c in enumerate(counts)}


@tf.function
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_brightness(image, 0.20)
    image = tf.image.random_contrast(image, 0.75, 1.25)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.08)
    crop = tf.random.uniform([], int(IMAGE_SIZE[0]*0.80), IMAGE_SIZE[0], dtype=tf.int32)
    oh = tf.random.uniform([], 0, IMAGE_SIZE[0]-crop, dtype=tf.int32)
    ow = tf.random.uniform([], 0, IMAGE_SIZE[1]-crop, dtype=tf.int32)
    image = image[oh:oh+crop, ow:ow+crop, :]
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image


def load_image_rgb(path):
    raw   = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image.numpy()


def extract_features(backbone, records, training=False, desc=""):
    n        = len(records)
    feat_dim = backbone.output_shape[-1]
    all_feats  = np.zeros((n, feat_dim), dtype=np.float32)
    all_labels = np.zeros(n, dtype=np.int32)

    print(f"  Extracting {desc} features: {n} samples (aug={training})...")
    t0 = time.time()

    for start in range(0, n, FEAT_BATCH):
        end   = min(start + FEAT_BATCH, n)
        batch = records[start:end]
        imgs  = np.zeros((end-start,)+IMAGE_SIZE+(3,), dtype=np.float32)
        for i, (path, _) in enumerate(batch):
            img = load_image_rgb(path)
            if training:
                img = augment_image(img)
                if hasattr(img, 'numpy'):
                    img = img.numpy()
            imgs[i] = img
        imgs_pp = preprocess_input(imgs)
        feats   = backbone.predict(imgs_pp, verbose=0)
        all_feats[start:end]  = feats
        all_labels[start:end] = [r[1] for r in batch]

        elapsed = time.time() - t0
        done = end / n
        eta  = (elapsed/done)*(1-done) if done > 0.001 else 0
        print(f"    {end:5d}/{n} ({done*100:.0f}%)  "
              f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m", end='\r', flush=True)
    print()
    return all_feats, all_labels


def build_head(feat_dim):
    return models.Sequential([
        layers.Input(shape=(feat_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.40),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.30),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.Dropout(0.20),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ], name='classification_head')


def assemble_model(backbone, head):
    inputs   = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    features = backbone(inputs, training=False)
    outputs  = head(features)
    return models.Model(inputs, outputs, name='SkinAI_EfficientNetV2B0')


def evaluate_per_class_feats(head, feats, labels):
    y_true = labels
    y_pred = np.argmax(head.predict(feats, batch_size=HEAD_BATCH, verbose=0), axis=1)
    overall = float((y_true == y_pred).mean())
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        per_class[name] = float((y_pred[mask]==i).mean()) if mask.sum()>0 else 0.0
    return overall, per_class


def make_ft_dataset(records, training=False):
    paths  = [r[0] for r in records]
    labels = [r[1] for r in records]

    @tf.function
    def load_pp(path, label):
        raw   = tf.io.read_file(path)
        image = tf.image.decode_image(raw, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = preprocess_input(image)
        return image, tf.one_hot(label, NUM_CLASSES)

    @tf.function
    def aug(image, label):
        img = (image+1.0)*127.5
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.80, 1.20)
        img = tf.image.rot90(img, tf.random.uniform([],0,4,dtype=tf.int32))
        img = tf.clip_by_value(img, 0.0, 255.0)
        return preprocess_input(img), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths),5000), seed=SEED)
    ds = ds.map(load_pp, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if training:
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(min(len(paths),2000))
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    return ds


class EpochLogger:
    def __init__(self, stage):
        self.stage = stage
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc  = logs.get('categorical_accuracy', logs.get('accuracy', 0))*100
        vacc = logs.get('val_categorical_accuracy', logs.get('val_accuracy', 0))*100
        loss = logs.get('loss', 0)
        top3 = logs.get('top3', 0)*100
        print(f"  [{self.stage}] Ep{epoch+1:3d} | loss={loss:.4f} | "
              f"acc={acc:.1f}% | val_acc={vacc:.1f}% | top3={top3:.1f}%")


def train():
    print("="*65)
    print("  SkinAI -- HAM10000 Feature-Extraction Training")
    print("="*65)

    img_map = list_images()
    print(f"  Found {len(img_map)} images on disk")
    records = load_records(img_map)
    print(f"  Loaded {len(records)} labelled records")

    raw_counts = defaultdict(int)
    for _, lbl in records: raw_counts[lbl] += 1
    print("\n  Raw class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:40s}: {raw_counts[i]:5d}")

    train_rec, val_rec, test_rec = stratified_split(records)
    train_rec = oversample(train_rec, MIN_PER_CLASS)

    over_counts = defaultdict(int)
    for _, lbl in train_rec: over_counts[lbl] += 1
    print(f"\n  After oversampling: train={len(train_rec)} | val={len(val_rec)} | test={len(test_rec)}")
    print("  Oversampled distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:40s}: {over_counts[i]:5d}")

    cw = class_weights_from_records(train_rec)
    print(f"\n  Class weights: { {CLASS_NAMES[k]:round(v,3) for k,v in cw.items()} }")

    print("\n  Loading EfficientNetV2B0 backbone (ImageNet weights, frozen)...")
    backbone = EfficientNetV2B0(
        include_top=False, weights='imagenet', pooling='avg',
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    backbone.trainable = False
    feat_dim = backbone.output_shape[-1]
    print(f"  Backbone output: {feat_dim}-dim feature vector")
    print(f"  Backbone params: {backbone.count_params():,}")

    cache_dir = os.path.join(PROJECT_ROOT, ".feat_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("\n  === STAGE 1: Feature Extraction ===")
    print("  Val/test features cached to disk (re-used on re-run).")
    print("  Training features also cached (delete .feat_cache/ to re-augment).")

    train_ff = os.path.join(cache_dir, "train_feats.npy")
    train_fl = os.path.join(cache_dir, "train_labels.npy")
    if os.path.exists(train_ff) and os.path.exists(train_fl):
        print("  Loading cached TRAIN features...")
        t_feats  = np.load(train_ff)
        t_labels = np.load(train_fl)
        if t_feats.shape[0] != len(train_rec):
            print(f"  Cache size mismatch ({t_feats.shape[0]} vs {len(train_rec)}), re-extracting...")
            t_feats, t_labels = extract_features(backbone, train_rec, training=True, desc="TRAIN")
            np.save(train_ff, t_feats); np.save(train_fl, t_labels)
    else:
        t_feats, t_labels = extract_features(backbone, train_rec, training=True, desc="TRAIN")
        np.save(train_ff, t_feats); np.save(train_fl, t_labels)

    vf_path = os.path.join(cache_dir, "val_feats.npy")
    vl_path = os.path.join(cache_dir, "val_labels.npy")
    if os.path.exists(vf_path):
        print("  Loading cached VAL features...")
        v_feats  = np.load(vf_path)
        v_labels = np.load(vl_path)
    else:
        v_feats, v_labels = extract_features(backbone, val_rec, training=False, desc="VAL")
        np.save(vf_path, v_feats); np.save(vl_path, v_labels)

    tf_path = os.path.join(cache_dir, "test_feats.npy")
    tl_path = os.path.join(cache_dir, "test_labels.npy")
    if os.path.exists(tf_path):
        print("  Loading cached TEST features...")
        te_feats  = np.load(tf_path)
        te_labels = np.load(tl_path)
    else:
        te_feats, te_labels = extract_features(backbone, test_rec, training=False, desc="TEST")
        np.save(tf_path, te_feats); np.save(tl_path, te_labels)

    print(f"\n  Feature shapes: train={t_feats.shape}, val={v_feats.shape}, test={te_feats.shape}")

    t_oh  = tf.keras.utils.to_categorical(t_labels,  NUM_CLASSES)
    v_oh  = tf.keras.utils.to_categorical(v_labels,  NUM_CLASSES)

    print("\n  === STAGE 2: Head Training ===")
    head = build_head(feat_dim)
    head.summary(print_fn=lambda x: print("  "+x))

    head.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=[CategoricalAccuracy(name='categorical_accuracy'),
                 TopKCategoricalAccuracy(k=3, name='top3')],
    )

    head_ckpt = os.path.join(PROJECT_ROOT, ".head_best.weights.h5")
    head.fit(
        t_feats, t_oh,
        validation_data=(v_feats, v_oh),
        epochs=HEAD_EPOCHS,
        batch_size=HEAD_BATCH,
        class_weight=cw,
        shuffle=True,
        callbacks=[
            ModelCheckpoint(head_ckpt, monitor='val_categorical_accuracy',
                            save_best_only=True, save_weights_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
                              min_lr=1e-6, verbose=1),
            EarlyStopping(monitor='val_categorical_accuracy', patience=20,
                          restore_best_weights=True, verbose=1),
            LambdaCallback(on_epoch_end=EpochLogger("Head").on_epoch_end),
        ],
        verbose=0,
    )
    head.load_weights(head_ckpt)

    s1_overall, s1_pca = evaluate_per_class_feats(head, te_feats, te_labels)
    print(f"\n  [After head training] Test acc: {s1_overall*100:.2f}%")
    for name, acc in s1_pca.items():
        bar = "#"*int(acc*30)
        print(f"    {name:40s}: {acc*100:5.1f}%  {bar}")

    # ── Evaluate on test features (fast, no image loading needed) ──────────
    print("\n  === Final Evaluation on Test Features ===")
    overall, per_class = evaluate_per_class_feats(head, te_feats, te_labels)

    # top-3 from head predictions
    y_pred_prob = head.predict(te_feats, batch_size=HEAD_BATCH, verbose=0)
    top3_correct = 0
    for i, true_lbl in enumerate(te_labels):
        top3 = np.argsort(y_pred_prob[i])[-3:]
        if true_lbl in top3:
            top3_correct += 1
    top3_acc = top3_correct / len(te_labels)

    print(f"  Overall accuracy : {overall*100:.2f}%")
    print(f"  Top-3  accuracy  : {top3_acc*100:.2f}%")
    print("\n  Per-class accuracy:")
    for name, acc in per_class.items():
        bar = "#"*int(acc*30)
        print(f"    {name:40s}: {acc*100:5.1f}%  {bar}")

    # ── Assemble + save final model (backbone frozen + trained head) ──────
    out_path = os.path.join(PROJECT_ROOT, "CNN1_trained_model.keras")
    print(f"\n  Assembling full inference model (backbone + head)...")
    full_model = assemble_model(backbone, head)
    full_model.save(out_path)
    print(f"  Saved --> {out_path}")

    # ── Save metadata and class names ─────────────────────────────────────
    with open(os.path.join(PROJECT_ROOT, "class_names.json"), "w") as fh:
        json.dump(CLASS_NAMES, fh, indent=2)

    metadata = {
        "image_size":    list(IMAGE_SIZE),
        "backbone":      "efficientnetv2b0",
        "dataset":       "HAM10000",
        "num_classes":   NUM_CLASSES,
        "class_names":   CLASS_NAMES,
        "preprocessing": "preprocess_input (efficientnet_v2) scales [0,255]->[-1,1]",
        "resize_method": "tf.image.resize bilinear",
        "training_strategy": "feature_extraction + head (backbone frozen)",
        "oversampling_min":  MIN_PER_CLASS,
        "label_smoothing":   LABEL_SMOOTH,
        "test_accuracy_pct":      round(overall*100, 2),
        "test_top3_accuracy_pct": round(top3_acc*100, 2),
        "per_class_accuracy_pct": {name: round(acc*100, 2) for name, acc in per_class.items()},
        "training_config": {
            "head_epochs": HEAD_EPOCHS,
            "feat_batch":  FEAT_BATCH,
            "head_batch":  HEAD_BATCH,
        },
    }
    with open(os.path.join(PROJECT_ROOT, "model_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"  Metadata --> model_metadata.json")
    print(f"  Classes  --> class_names.json")
    print("="*65)
    print("  Training complete!")
    print("="*65)

    # ── Optional Stage 3: Fine-tune (set DO_FINETUNE=1 to enable) ────────
    # Skipped by default — on CPU each fine-tune epoch takes ~25+ minutes.
    # Once you have a GPU available, re-run with: DO_FINETUNE=1 python train.py
    do_finetune = os.getenv("DO_FINETUNE", "0").strip() == "1"
    if not do_finetune:
        print("\n  [Stage 3 skipped] Set DO_FINETUNE=1 to run backbone fine-tuning.")
        return

    print(f"\n  === STAGE 3: Fine-tune last 30 backbone layers ({FT_EPOCHS} epochs) ===")
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    print(f"  Trainable backbone layers: {sum(1 for l in backbone.layers if l.trainable)}")

    full_model.compile(
        optimizer=AdamW(learning_rate=FT_LR, weight_decay=1e-5),
        loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=[CategoricalAccuracy(name='accuracy'),
                 TopKCategoricalAccuracy(k=3, name='top3')],
    )

    # Small balanced subset so fine-tune stays manageable
    ft_train_recs = oversample(train_rec, min_per_class=min(600, MIN_PER_CLASS))
    ft_ds  = make_ft_dataset(ft_train_recs, training=True)
    val_ds = make_ft_dataset(val_rec,       training=False)

    ft_out = os.path.join(PROJECT_ROOT, "CNN1_trained_model.keras")
    full_model.fit(
        ft_ds, validation_data=val_ds,
        epochs=FT_EPOCHS, class_weight=cw,
        callbacks=[
            ModelCheckpoint(ft_out, monitor='val_accuracy',
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3,
                              min_lr=1e-8, verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=5,
                          restore_best_weights=True, verbose=1),
            LambdaCallback(on_epoch_end=EpochLogger("FineTune").on_epoch_end),
        ],
        verbose=0,
    )
    print(f"  Fine-tuned model --> {ft_out}")
    print("="*65)
    print("  Fine-tuning complete!")
    print("="*65)


if __name__ == "__main__":
    train()
