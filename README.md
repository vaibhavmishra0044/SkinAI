# SkinAI - AI Skin Condition Analysis

SkinAI is a Flask web app for AI-assisted skin image analysis. It accepts a clear close-up skin photo, checks whether a visible lesion or rash is present, then returns a suspected condition, top alternative matches, visible signs, safety guidance, and next steps.

This project is not a medical diagnostic device. Results are educational screening support only and should be confirmed by a qualified dermatologist.

## Current Analysis Flow

1. User uploads a JPG, PNG, or WebP skin image, or captures one with the device camera.
2. The app rejects images that are too small, too wide, unclear, or not focused on an affected skin area.
3. If `NVIDIA_API_KEY` is configured, the app uses AI vision analysis as the primary analyzer.
4. The AI vision workflow can return a broad suspected dermatology condition name when the image supports it.
5. If the AI is uncertain, the app marks the result as low confidence and shows safer next steps instead of forcing a disease name.
6. If the API is unavailable, the app falls back to the limited local model and labels that result as a fallback.

## What It Can Analyze

The API-first workflow can reason over many visible skin-condition patterns, including:

- Inflammatory rashes and dermatitis-like conditions
- Fungal, bacterial, viral, and parasitic patterns
- Acne and follicular disorders
- Pigment changes
- Moles, growths, and keratosis-like lesions
- Vascular-looking lesions
- Possible skin cancer warning signs

No AI photo tool can honestly detect every skin disease with perfect accuracy. Some diseases require history, touch, dermoscopy, lab tests, biopsy, or in-person examination.

## Environment Variables

Create a local `.env` file or configure these in your hosting provider:

```bash
NVIDIA_API_KEY=your_key_here
NVIDIA_MODEL=nvidia/nemotron-nano-12b-v2-vl
SECRET_KEY=change_me
DATABASE_URL=sqlite:///site.db
```

If `NVIDIA_API_KEY` is missing, the app uses the local fallback analyzer.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Deploy

This repo includes Vercel configuration:

- `vercel.json`
- `api/index.py`
- `.vercelignore`

Deploy with:

```bash
vercel deploy --prod
```

Set `NVIDIA_API_KEY` and `NVIDIA_MODEL` in Vercel environment variables before deploying.

## Local Fallback Model

The repository still includes the older local model files:

- `CNN1_trained_model.tflite`
- `CNN1_trained_model.keras`
- `class_names.json`

These are used only as a fallback when API vision analysis is unavailable. They are limited and should not be presented as broad disease detection.

For retraining experiments, install:

```bash
pip install -r requirements-train.txt
```

Then run:

```bash
python train.py
```
