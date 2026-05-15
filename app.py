import os, json, secrets, tempfile, datetime
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
# Keep HF model weights on E: (C: has limited space)
os.environ.setdefault('HF_HOME', r'E:\hf_cache')
os.environ.setdefault('TRANSFORMERS_CACHE', r'E:\hf_cache\hub')
import numpy as np
from functools import wraps
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

try:
    from ai_edge_litert.interpreter import Interpreter
    _TFLITE_OK = True
except ImportError:
    _TFLITE_OK = False

try:
    from transformers import pipeline as _hf_pipeline
    _HF_TRANSFORMERS_OK = True
except ImportError:
    _HF_TRANSFORMERS_OK = False

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or secrets.token_hex(32) # SQL ALCEHMY SECTRETE TOCKEN HEX 32 CONFIGER MANAGEMNET OF OGINN OF ID OF THE 
if os.getenv('VERCEL'):
    db_path = os.path.join(tempfile.gettempdir(), 'site.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path.replace('\\', '/')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # TRUE VALUE = 1000 AAND FIX THE NOTEBOOK 
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'skinai_uploads') if os.getenv('VERCEL') else os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
# APP.SECRETE KEY TOKEN = SQLITE DATABASE AKLTERNATIVE AND FIX THE ALGORITHNEM 
#APP.CONFGURATIOM { FALSE- SWITH THE MMODE MAX_CONTENT LEGTH =10 MEGABITE AND FIX THE NOTE BBOOK}
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# ── Jinja2 Globals & Filters ──────────────────────────────────────────────────
app.jinja_env.globals['now'] = datetime.datetime.now
app.jinja_env.filters['from_json'] = json.loads

# ── Load CNN Model ────────────────────────────────────────────────────────────
tflite_model_path = os.path.join(app.root_path, 'CNN1_trained_model.tflite')
keras_model_path = os.path.join(app.root_path, 'CNN1_trained_model.keras')
cnn_model = None
_tflite_interpreter = None
_tflite_input_details = None
_tflite_output_details = None
CNN_RUNTIME = None

if _TFLITE_OK and os.path.exists(tflite_model_path):
    _tflite_interpreter = Interpreter(model_path=tflite_model_path)
    _tflite_interpreter.allocate_tensors()
    _tflite_input_details = _tflite_interpreter.get_input_details()
    _tflite_output_details = _tflite_interpreter.get_output_details()
    CNN_RUNTIME = "tflite"
else:
    from tensorflow.keras.models import load_model
    cnn_model = load_model(keras_model_path)
    CNN_RUNTIME = "keras"

IMAGE_SIZE = (224, 224)

with open(os.path.join(app.root_path, 'class_names.json'), 'r') as f:
    CLASS_NAMES = json.load(f)

DISPLAY_NAMES = {
    # LaurianeMD 22-class ViT model (HuggingFace)
    "acne":                         "Acne",
    "actinic_keratosis":            "Actinic Keratosis",
    "benign_tumors":                "Benign Tumors",
    "bullous":                      "Bullous Disease",
    "candidiasis":                  "Candidiasis (Fungal)",
    "drug_eruption":                "Drug Eruption",
    "eczema":                       "Eczema",
    "infestations_bites":           "Infestations / Insect Bites",
    "lichen":                       "Lichen Planus",
    "lupus":                        "Lupus / Connective Tissue Disease",
    "moles":                        "Moles / Melanocytic Nevi",
    "psoriasis":                    "Psoriasis",
    "rosacea":                      "Rosacea",
    "seborrheic_keratoses":         "Seborrheic Keratoses",
    "skin_cancer":                  "Skin Cancer",
    "sun_damage":                   "Sun / Sunlight Damage",
    "tinea_ringworm":               "Tinea / Ringworm / Fungal Infection",
    "unknown_normal":               "Normal Skin",
    "vascular_tumors":              "Vascular Tumors",
    "vasculitis":                   "Vasculitis",
    "vitiligo":                     "Vitiligo",
    "warts":                        "Warts / Viral Infection",

    # Local HAM10000 CNN model classes
    "actinic_keratoses":             "Actinic Keratoses",
    "basal_cell_carcinoma":          "Basal Cell Carcinoma",
    "benign_keratosis_like_lesions": "Benign Keratosis-like Lesions",
    "dermatofibroma":                "Dermatofibroma",
    "melanoma":                      "Melanoma",
    "melanocytic_nevi":              "Melanocytic Nevi / Moles",
    "vascular_lesions":              "Vascular Lesions",
}

LESION_INFO = {
    # ── LaurianeMD 22-class ViT model ────────────────────────────────────────
    "tinea_ringworm": {
        "description": "Fungal skin infection causing ring-shaped, red, scaly, itchy patches. Ringworm (Tinea corporis) is not caused by a worm — it's a dermatophyte fungus. Highly contagious and very treatable.",
        "cause": "Dermatophyte fungi (Trichophyton, Microsporum, Epidermophyton). Spread through direct contact with infected people, animals, soil, or contaminated surfaces/clothing.",
        "recommendations": "Apply OTC antifungal cream (clotrimazole, terbinafine, miconazole) 2x daily for 2-4 weeks. For scalp or widespread infections, see a doctor for oral antifungals (fluconazole, griseofulvin).",
        "risk_level": "Moderate",
        "precautions": "Keep affected area clean and dry. Do not share towels, clothing, or combs. Wash hands after touching the area. Complete the full treatment course even if symptoms disappear.",
        "icd": "B35.4"
    },
    "acne": {
        "description": "Acne is a very common skin condition involving blocked pores, whiteheads, blackheads, pimples, and cysts. Affects face, chest, and back. Usually begins in puberty but can affect all ages.",
        "cause": "Excess sebum (oil) production, Cutibacterium acnes bacteria, clogged hair follicles, and hormonal changes. Stress and certain foods may worsen it.",
        "recommendations": "Mild: OTC benzoyl peroxide, salicylic acid, or adapalene. Moderate-severe: prescription retinoids, antibiotics (topical or oral), or isotretinoin for cystic acne.",
        "risk_level": "Low",
        "precautions": "Avoid picking or squeezing pimples. Wash face twice daily with a gentle cleanser. Use non-comedogenic skincare products. Apply SPF 30+ sunscreen daily.",
        "icd": "L70.0"
    },
    "actinic_keratosis": {
        "description": "Rough, scaly patches on sun-damaged skin. Pre-cancerous lesion that can progress to squamous cell carcinoma. Most common in people over 40 with significant sun exposure history.",
        "cause": "Chronic UV radiation exposure from sun or tanning beds. Immunosuppressed individuals are at higher risk.",
        "recommendations": "Consult a dermatologist promptly. Treatment: cryotherapy, topical 5-FU cream, imiquimod, photodynamic therapy, or diclofenac gel.",
        "risk_level": "Moderate",
        "precautions": "Apply SPF 50+ sunscreen daily. Wear hats and sun-protective clothing. Schedule regular skin checks. Avoid tanning beds.",
        "icd": "L57.0"
    },
    "benign_tumors": {
        "description": "Non-cancerous skin growths including seborrheic keratoses, dermatofibromas, lipomas, and epidermoid cysts. Very common, especially in older adults. Generally harmless.",
        "cause": "Aging, genetics, sun exposure (for some types). Not infectious and not contagious. Generally no malignant potential.",
        "recommendations": "Usually no treatment needed. Removal available for cosmetic reasons or if causing discomfort: cryotherapy, excision, or laser therapy.",
        "risk_level": "Low",
        "precautions": "Monitor for any rapid changes in size, shape, or color that could signal malignancy. See a dermatologist if a lesion changes quickly.",
        "icd": "D23"
    },
    "bullous": {
        "description": "A group of conditions causing large fluid-filled blisters (bullae) on the skin and/or mucous membranes. Includes pemphigus vulgaris and bullous pemphigoid.",
        "cause": "Autoimmune conditions where the immune system attacks proteins holding skin cells together. Some cases triggered by medications.",
        "recommendations": "Requires urgent specialist evaluation. Treatment involves systemic corticosteroids and immunosuppressants. Do not rupture blisters without medical guidance.",
        "risk_level": "High",
        "precautions": "Seek immediate medical care. Keep blisters intact if possible. Avoid friction and trauma to affected areas. Monitor for signs of secondary infection.",
        "icd": "L12"
    },
    "candidiasis": {
        "description": "Fungal infection caused by Candida yeast. Causes red, itchy rash with satellite pustules — commonly in skin folds, groin, armpits, or under the breast. Also causes thrush (mouth) and vaginal infections.",
        "cause": "Overgrowth of Candida albicans yeast. Risk factors: diabetes, antibiotic use, immunosuppression, obesity, excessive moisture/sweating.",
        "recommendations": "Topical antifungal creams (clotrimazole, miconazole, nystatin) applied 2x daily. Keep skin folds dry. For recurring cases or oral/genital candidiasis, oral fluconazole may be prescribed.",
        "risk_level": "Moderate",
        "precautions": "Keep skin dry and clean, especially in folds. Change damp clothing promptly. Manage blood sugar if diabetic. Use loose, breathable fabrics.",
        "icd": "B37.2"
    },
    "drug_eruption": {
        "description": "Skin reactions caused by medications, ranging from mild rashes to life-threatening conditions like Stevens-Johnson Syndrome (SJS). Any drug can potentially cause a reaction.",
        "cause": "Immune-mediated or direct toxic reaction to a medication. Common culprits: antibiotics, NSAIDs, anticonvulsants, and chemotherapy agents.",
        "recommendations": "Stop the suspected medication immediately and consult a doctor. Mild: antihistamines, topical steroids. Severe reactions (SJS/TEN): emergency hospitalization required.",
        "risk_level": "High",
        "precautions": "STOP the suspected medication and seek medical care immediately. Do not restart without medical advice. Carry a list of all drug allergies.",
        "icd": "L27.0"
    },
    "infestations_bites": {
        "description": "Skin conditions caused by mites, lice, fleas, bedbugs, or insect bites. Presents as intensely itchy rashes, small bumps, or burrow tracks. Scabies mites are a common cause.",
        "cause": "Sarcoptes scabiei (scabies), Pediculus (lice), flea or bedbug bites, or other parasites. Spread through close skin contact or infested bedding/clothing.",
        "recommendations": "Scabies: prescription permethrin 5% cream (applied overnight, repeat after 1 week) or oral ivermectin. Treat all household contacts simultaneously. Wash all bedding and clothing in hot water.",
        "risk_level": "Moderate",
        "precautions": "Wash all clothing, bedding, and towels in hot water (60°C+). Seal non-washable items in bags for 72 hours. Avoid sharing clothing. Treat all close contacts simultaneously.",
        "icd": "B86"
    },
    "lichen": {
        "description": "Lichen planus is an inflammatory condition causing flat-topped, purple, itchy bumps on skin and white lacy patches in the mouth. Can also affect nails and scalp.",
        "cause": "Autoimmune reaction — T-cells attack skin or mucosal cells. Triggered by hepatitis C infection, certain medications, or dental materials. Exact cause often unknown.",
        "recommendations": "Mild: topical corticosteroids. Moderate-severe: oral corticosteroids, retinoids, cyclosporine, or phototherapy. Oral lichen planus may need antifungals if secondary candidiasis develops.",
        "risk_level": "Moderate",
        "precautions": "Avoid known triggers. Protect affected skin from friction and trauma. Oral lesions need regular monitoring — rarely can progress to oral cancer. Maintain good dental hygiene.",
        "icd": "L43"
    },
    "moles": {
        "description": "Moles (melanocytic nevi) are benign growths of melanocytes. Most adults have 10-40 moles. They can be flat or raised, vary in color, and appear anywhere on the body.",
        "cause": "Clusters of melanocyte cells. Can be congenital (present at birth) or appear during childhood and adolescence. UV exposure can increase number and darken existing moles.",
        "recommendations": "Regular self-monitoring using the ABCDE rule: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving. Annual dermatologist check recommended. Biopsy any suspicious mole.",
        "risk_level": "Low",
        "precautions": "Perform monthly skin self-exams. Protect moles from UV exposure — use SPF 50+. See a dermatologist immediately if any mole changes, bleeds, or looks unusual.",
        "icd": "D22"
    },
    "rosacea": {
        "description": "A chronic skin condition causing facial redness, visible blood vessels (telangiectasia), and sometimes acne-like bumps. Primarily affects the cheeks, nose, and forehead.",
        "cause": "Exact cause unknown. Contributing factors: abnormal immune response, skin microbiome changes, sun exposure, genetics. Triggers: heat, spicy food, alcohol, stress, exercise.",
        "recommendations": "Topical metronidazole, azelaic acid, or ivermectin cream. Oral antibiotics for inflammatory bumps. Laser therapy for persistent redness. Identify and avoid personal triggers.",
        "risk_level": "Low",
        "precautions": "Apply SPF 30+ sunscreen daily. Avoid known triggers. Use gentle, fragrance-free skincare. Do not use steroid creams (may worsen rosacea). See a dermatologist for proper diagnosis.",
        "icd": "L71"
    },
    "seborrheic_keratoses": {
        "description": "Common, non-cancerous growths appearing as waxy, stuck-on, brown or black patches. Very common in adults over 40. Can look alarming but are entirely benign.",
        "cause": "Exact cause unknown. Associated with aging and possibly sun exposure. Strong genetic predisposition. Not contagious and not pre-cancerous.",
        "recommendations": "No treatment necessary unless cosmetically bothersome or irritated. Removal options: cryotherapy, electrocautery, curettage, or laser. Always confirm benign diagnosis first.",
        "risk_level": "Low",
        "precautions": "Monitor for rapid changes in appearance (sudden increase in size, bleeding, or change in color) that may indicate a different condition. See a dermatologist if a lesion changes quickly.",
        "icd": "L82"
    },
    "skin_cancer": {
        "description": "Malignant tumors of the skin including basal cell carcinoma (BCC), squamous cell carcinoma (SCC), and melanoma. Most are caused by UV radiation and are highly treatable when caught early.",
        "cause": "Chronic UV radiation (sun, tanning beds), fair skin, family history, immunosuppression, HPV infection (for SCC), or radiation therapy history.",
        "recommendations": "URGENT: See a dermatologist immediately. Treatment depends on type and stage: surgical excision, Mohs surgery, cryotherapy, radiation, chemotherapy, immunotherapy, or targeted therapy.",
        "risk_level": "Very High",
        "precautions": "SEEK IMMEDIATE MEDICAL EVALUATION — do not delay. Apply SPF 50+ sunscreen daily. Wear protective clothing. Perform regular self-exams. Annual full-body skin checks.",
        "icd": "C44"
    },
    "sun_damage": {
        "description": "Skin damage from prolonged UV radiation exposure, including sunburn, photoaging (wrinkles, brown spots), freckles, and pre-cancerous changes. Cumulative UV damage is the main cause of skin cancer.",
        "cause": "Ultraviolet (UV) radiation from the sun or tanning beds damaging DNA in skin cells and breaking down collagen. Both UVA (aging) and UVB (burning) rays cause damage.",
        "recommendations": "Use SPF 50+ broad-spectrum sunscreen daily. Topical retinoids for photoaging. Chemical peels or laser for brown spots. See a dermatologist to check for pre-cancerous lesions.",
        "risk_level": "Moderate",
        "precautions": "Apply SPF 50+ sunscreen every 2 hours when outdoors. Wear UV-protective clothing, hats, and sunglasses. Avoid peak UV hours (10am-4pm). Never use tanning beds.",
        "icd": "L57.8"
    },
    "unknown_normal": {
        "description": "The image appears to show normal skin or a condition the model cannot confidently classify. No specific skin disease pattern was detected with high confidence.",
        "cause": "Could be normal skin, a very mild skin condition, or an image that doesn't match the training distribution of the AI model.",
        "recommendations": "If you have any skin concerns, always consult a qualified dermatologist for a proper in-person evaluation. Do not rely solely on AI for medical diagnosis.",
        "risk_level": "Low",
        "precautions": "Maintain good general skin hygiene. Apply sunscreen daily. See a dermatologist if you have any persistent skin symptoms, changes, or concerns.",
        "icd": "Z00.0"
    },
    "vitiligo": {
        "description": "A chronic autoimmune condition where the immune system destroys melanocytes (pigment-producing cells), causing white patches of depigmentation on skin, hair, or inside the mouth.",
        "cause": "Autoimmune destruction of melanocytes. Genetic predisposition plus environmental triggers. Associated with other autoimmune conditions (thyroid disease, type 1 diabetes).",
        "recommendations": "Topical corticosteroids or calcineurin inhibitors. Phototherapy (narrowband UVB) for widespread cases. Ruxolitinib cream (JAK inhibitor) — newer targeted option. Cosmetic camouflage.",
        "risk_level": "Low",
        "precautions": "Apply SPF 50+ sunscreen to depigmented areas — they burn very easily. Check thyroid function annually. Psychological support may be needed as vitiligo significantly affects quality of life.",
        "icd": "L80"
    },
    "warts": {
        "description": "Viral skin growths caused by human papillomavirus (HPV). Include common warts, plantar warts (feet), flat warts, and filiform warts. Usually benign and self-limiting but contagious.",
        "cause": "Human papillomavirus (HPV). Spread through direct skin contact or contaminated surfaces. Risk increased by skin abrasions, weakened immune system, or barefoot walking in public.",
        "recommendations": "Many warts resolve spontaneously within 1-2 years. Treatment: salicylic acid (OTC), cryotherapy, electrocautery, laser therapy, or topical immunotherapy (imiquimod).",
        "risk_level": "Low",
        "precautions": "Cover warts with a bandage. Avoid touching or picking. Wear flip-flops in communal showers. Do not share personal items. HPV vaccine prevents genital warts and related cancers.",
        "icd": "B07"
    },
    "eczema": {
        "description": "A chronic skin condition causing itchy, inflamed, red, and cracked patches. Closely related to atopic dermatitis. Very common, affecting both children and adults.",
        "cause": "Combination of genetic skin barrier defects and immune dysregulation. Triggered by irritants, allergens, stress, sweat, and weather changes.",
        "recommendations": "Regular moisturizing is essential. Topical corticosteroids for flares. Avoid known triggers. For severe cases: topical tacrolimus, dupilumab, or phototherapy.",
        "risk_level": "Low",
        "precautions": "Use fragrance-free, hypoallergenic products. Keep skin hydrated. Avoid hot showers. Wear cotton clothing. Identify and avoid personal triggers.",
        "icd": "L30.9"
    },
    "lupus": {
        "description": "Systemic lupus erythematosus (SLE) and cutaneous lupus can cause distinctive skin rashes including the butterfly rash across the nose and cheeks, disc-shaped patches, and photosensitivity.",
        "cause": "Autoimmune condition where the immune system attacks healthy tissue. Genetic predisposition with environmental triggers (UV light, infections, stress).",
        "recommendations": "Requires rheumatologist/dermatologist evaluation. Treatment: antimalarials (hydroxychloroquine), corticosteroids, immunosuppressants. Strict sun avoidance is critical.",
        "risk_level": "High",
        "precautions": "Avoid sun exposure — use SPF 50+ and protective clothing. Do not stop medications without medical advice. Monitor for organ involvement (kidneys, heart, lungs).",
        "icd": "L93"
    },
    "psoriasis": {
        "description": "A chronic autoimmune skin condition causing rapid buildup of skin cells, resulting in silvery scales, red patches, and dry, cracked skin that may bleed. Often affects elbows, knees, and scalp.",
        "cause": "Autoimmune — T-cells attack healthy skin cells. Triggers: stress, infections, certain medications, skin injuries, smoking, and alcohol.",
        "recommendations": "Topical treatments: corticosteroids, vitamin D analogues, retinoids. Phototherapy. Systemic medications: methotrexate, cyclosporine. Biologics for severe cases (TNF-alpha inhibitors).",
        "risk_level": "Moderate",
        "precautions": "Moisturize daily. Avoid scratching and picking. Identify and manage triggers (stress, infections). Protect skin from injury. Some patients may develop psoriatic arthritis.",
        "icd": "L40"
    },
    "vascular_tumors": {
        "description": "Tumors arising from blood or lymphatic vessels, ranging from benign hemangiomas and port-wine stains to malignant angiosarcomas. Most are benign.",
        "cause": "Abnormal proliferation of vascular endothelial cells. Some are congenital; others develop with age or after trauma or radiation.",
        "recommendations": "Most require no treatment. Large or symptomatic lesions: laser therapy, sclerotherapy, or surgery. Any rapidly growing vascular lesion requires urgent evaluation to rule out malignancy.",
        "risk_level": "Moderate",
        "precautions": "Monitor for rapid growth, bleeding, or ulceration. Facial or large lesions in infants should be evaluated by a specialist. Avoid trauma to the affected area.",
        "icd": "D18.0"
    },
    "vasculitis": {
        "description": "Inflammation of blood vessel walls causing skin rash (palpable purpura, ulcers) and potentially affecting internal organs. A serious condition requiring medical evaluation.",
        "cause": "Autoimmune conditions, infections, medications, or unknown causes triggering immune-mediated vessel inflammation.",
        "recommendations": "Requires urgent medical evaluation. Treatment depends on severity and underlying cause: corticosteroids, immunosuppressants, or treatment of the underlying trigger.",
        "risk_level": "High",
        "precautions": "Seek prompt medical care. Do not ignore purplish spots or skin ulcers that don't heal. Monitor for organ involvement (kidneys, lungs). Avoid triggers like certain medications.",
        "icd": "M31.9"
    },
}

INFO_ALIASES = {
    "actinic_keratoses": "actinic_keratosis",
    "basal_cell_carcinoma": "skin_cancer",
    "benign_keratosis_like_lesions": "benign_tumors",
    "dermatofibroma": "benign_tumors",
    "melanoma": "skin_cancer",
    "melanocytic_nevi": "moles",
    "vascular_lesions": "vascular_tumors",
}

RISK_COLORS = {
    "Low":       ("green",  "#16a34a", "#dcfce7"),
    "Moderate":  ("yellow", "#d97706", "#fef3c7"),
    "High":      ("orange", "#ea580c", "#ffedd5"),
    "Very High": ("red",    "#dc2626", "#fee2e2"),
}

HF_LABEL_MAP = {
    # LaurianeMD 22-class model (primary HF model)
    "acne":                   "acne",
    "actinic keratosis":      "actinic_keratosis",
    "benign tumors":          "benign_tumors",
    "bullous":                "bullous",
    "candidiasis":            "candidiasis",
    "drug eruption":          "drug_eruption",
    "eczema":                 "eczema",
    "infestations bites":     "infestations_bites",
    "lichen":                 "lichen",
    "lupus":                  "lupus",
    "moles":                  "moles",
    "psoriasis":              "psoriasis",
    "rosacea":                "rosacea",
    "seborrh keratoses":      "seborrheic_keratoses",
    "skin cancer":            "skin_cancer",
    "sun sunlight damage":    "sun_damage",
    "tinea":                  "tinea_ringworm",
    "unknown normal":         "unknown_normal",
    "vascular tumors":        "vascular_tumors",
    "vasculitis":             "vasculitis",
    "vitiligo":               "vitiligo",
    "warts":                  "warts",
}

def normalize_hf_label(label_str):
    key = HF_LABEL_MAP.get(label_str.strip().lower())
    if key:
        return key
    return label_str.lower().replace(' ', '_').replace('-', '_').replace('/', '_')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Resize to the model input and scale pixels to the EfficientNetV2 range."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)               # (1, H, W, 3)
# DEF ALLOWED_FILE(FILE NAME ) DEIGN NURAL NETWORK AND HUMAIN BRAIN NP ZERO (1,IMAGE SIZE {00=1173,3445,16555,8867,})

# ── HuggingFace Local Pipeline (23-class DermNet model) ───────────────────────
# Anwarkh1 covers ringworm/tinea, eczema, psoriasis, acne + 19 other conditions
# Weights download once to E:\hf_cache (~90 MB) on first startup
HF_MODEL_ID = "LaurianeMD/vit-skin-disease"
_hf_clf = None
_HF_ENABLED = os.getenv('SKINAI_ENABLE_HF', '0' if os.getenv('VERCEL') else '1').strip().lower() in {'1', 'true', 'yes', 'on'}

if _HF_TRANSFORMERS_OK and _HF_ENABLED:
    try:
        print(f"[SkinAI] Loading HuggingFace model '{HF_MODEL_ID}'...")
        _hf_clf = _hf_pipeline(
            task="image-classification",
            model=HF_MODEL_ID,
            top_k=None,
            device=-1,
        )
        print(f"[SkinAI] HF model loaded — 22-class skin disease detection active.")
    except Exception as _e:
        print(f"[SkinAI] Could not load HF model: {_e}")
        _hf_clf = None
elif _HF_TRANSFORMERS_OK:
    print("[SkinAI] HuggingFace model disabled; using local CNN model.")
else:
    print("[SkinAI] transformers not installed.")

def run_hf_inference(image_path):
    results = _hf_clf(image_path)   # list of {label, score}
    score_map = {}
    for r in results:
        key = normalize_hf_label(r['label'])
        if key not in score_map or r['score'] > score_map[key]:
            score_map[key] = r['score']
    total = sum(score_map.values()) or 1.0
    score_map = {k: v / total for k, v in score_map.items()}
    sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [{'class_key': k, 'confidence': round(v * 100, 1)} for k, v in sorted_items[:3]]

def run_cnn_inference(image_path):
    input_data = preprocess_image(image_path)
    if CNN_RUNTIME == "tflite":
        input_detail = _tflite_input_details[0]
        output_detail = _tflite_output_details[0]
        if input_data.dtype != input_detail['dtype']:
            input_data = input_data.astype(input_detail['dtype'])
        _tflite_interpreter.set_tensor(input_detail['index'], input_data)
        _tflite_interpreter.invoke()
        preds = _tflite_interpreter.get_tensor(output_detail['index'])[0]
    else:
        preds = cnn_model.predict(input_data, verbose=0)[0]
    preds = np.asarray(preds, dtype=np.float32)

    # Accept either softmax probabilities or raw logits from the model.
    if np.any(preds < 0) or not np.isclose(float(np.sum(preds)), 1.0, atol=1e-3):
        preds = np.exp(preds - np.max(preds))
        preds = preds / (np.sum(preds) or 1.0)

    top_indices = np.argsort(preds)[::-1][:3]
    return [
        {
            'class_key': CLASS_NAMES[int(i)],
            'confidence': round(float(preds[int(i)]) * 100, 1),
        }
        for i in top_indices
    ]

# ── Database Models ───────────────────────────────────────────────────────────
class User(db.Model, UserMixin):
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(150), unique=True, nullable=False)
    email      = db.Column(db.String(150), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)
    is_admin   = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    feedbacks  = db.relationship('Feedback',   backref='author', lazy=True)
    predictions= db.relationship('Prediction', backref='user',   lazy=True)

class Doctor(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    name           = db.Column(db.String(150), nullable=False)
    specialization = db.Column(db.String(150), nullable=False)
    phone          = db.Column(db.String(50),  nullable=False)
    email          = db.Column(db.String(150), default='')
    hospital       = db.Column(db.String(200), default='')
    available      = db.Column(db.Boolean, default=True)

class Feedback(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    content    = db.Column(db.Text, nullable=False)
    rating     = db.Column(db.Integer, default=5)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
#HOSPITAL = DB.COLUMN (DB STRIG 15-84)
class Prediction(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    predicted_class = db.Column(db.String(150))
    predicted_key   = db.Column(db.String(100))
    confidence      = db.Column(db.Float)
    risk_level      = db.Column(db.String(50))
    top3_json       = db.Column(db.Text)
    created_at      = db.Column(db.DateTime, default=datetime.datetime.utcnow)
# PARAMETER = DB COLOUM AND FIND THE RESISTANCE 992
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_admin:
            flash('Admin access required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated

# ── Auth Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard') if current_user.is_admin else url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        db.session.add(User(username=username, email=email, password=generate_password_hash(password)))
        db.session.commit()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login(): # LOGIN THE FOLOWING 98_12{ 12# 87}
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('admin_dashboard') if user.is_admin else url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ── User Routes ───────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    doctors      = Doctor.query.filter_by(available=True).all()
    recent_preds = (Prediction.query.filter_by(user_id=current_user.id)
                    .order_by(Prediction.created_at.desc()).limit(5).all())
    total_preds  = Prediction.query.filter_by(user_id=current_user.id).count()
    return render_template('dashboard.html',
        doctors=doctors, recent_preds=recent_preds, total_preds=total_preds,
        now=datetime.datetime.now)

@app.route('/history')
@login_required
def history():
    preds = (Prediction.query.filter_by(user_id=current_user.id)
             .order_by(Prediction.created_at.desc()).all())
    return render_template('history.html', preds=preds)

@app.route('/detect')
@login_required
def detect_page():
    return render_template('detect.html')

@app.route('/scan')
def scan_page():
    return render_template('scan_window.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG, PNG, or WebP.'}), 400

    ext = os.path.splitext(secure_filename(file.filename))[1].lower() or '.jpg'
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        file.save(tmp.name)
        tmp.close()
        if _hf_clf is not None:
            top3_raw = run_hf_inference(tmp.name)
            model_used = "ViT 22-class (HuggingFace)"
        else:
            top3_raw = run_cnn_inference(tmp.name)
            model_used = "CNN 7-class (HAM10000)"
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    top3 = []
    for item in top3_raw:
        key = item['class_key']
        top3.append({
            'class_key':  key,
            'class_name': DISPLAY_NAMES.get(key, key.replace('_', ' ').title()),
            'confidence': item['confidence'],
        })

    primary      = top3[0]
    predicted_key = primary['class_key']
    info_key = INFO_ALIASES.get(predicted_key, predicted_key)
    info = LESION_INFO.get(info_key, {
        'description':     'Information not available.',
        'cause':           'Unknown.',
        'recommendations': 'Consult a dermatologist.',
        'risk_level':      'Unknown',
        'precautions':     'Use sun protection and consult a dermatologist.',
        'icd':             'N/A',
    })
    risk_info = RISK_COLORS.get(info['risk_level'], ('gray', '#64748b', '#f1f5f9'))

    # Save prediction to DB only for authenticated users
    if current_user.is_authenticated:
        pred_record = Prediction(
            user_id         = current_user.id,
            predicted_class = primary['class_name'],
            predicted_key   = predicted_key,
            confidence      = primary['confidence'],
            risk_level      = info['risk_level'],
            top3_json       = json.dumps(top3),
        )
        db.session.add(pred_record)
        db.session.commit()

    return jsonify({
        'predicted_class':     primary['class_name'],
        'predicted_class_key': predicted_key,
        'confidence':          primary['confidence'],
        'top3':                top3,
        'description':         info['description'],
        'cause':               info['cause'],
        'recommendations':     info['recommendations'],
        'risk_level':          info['risk_level'],
        'precautions':         info['precautions'],
        'icd':                 info.get('icd', 'N/A'),
        'risk_color':          risk_info[1],
        'risk_bg':             risk_info[2],
        'model_used':          model_used,
    })

@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    content = request.form.get('content', '').strip()
    try:
        rating = max(1, min(5, int(request.form.get('rating', 5))))
    except (ValueError, TypeError):
        rating = 5
    if not content:
        flash('Feedback cannot be empty.', 'danger')
        return redirect(url_for('dashboard'))
    db.session.add(Feedback(content=content, rating=rating, user_id=current_user.id))
    db.session.commit()
    flash('Thank you for your feedback!', 'success')
    return redirect(url_for('dashboard'))

# ── Admin Routes ──────────────────────────────────────────────────────────────
@app.route('/admin')
@admin_required
def admin_dashboard():
    doctors     = Doctor.query.all()
    users       = User.query.order_by(User.created_at.desc()).all()
    feedbacks   = (Feedback.query.order_by(Feedback.created_at.desc()).all())
    predictions = (Prediction.query.order_by(Prediction.created_at.desc()).limit(100).all())

    total_users       = User.query.filter_by(is_admin=False).count()
    total_doctors     = Doctor.query.count()
    total_predictions = Prediction.query.count()
    total_feedback    = Feedback.query.count()

    # Prediction distribution for chart
    class_counts = {}
    for p in Prediction.query.all():
        class_counts[p.predicted_class] = class_counts.get(p.predicted_class, 0) + 1

    return render_template('admin_dashboard.html',
        doctors=doctors, users=users, feedbacks=feedbacks, predictions=predictions,
        total_users=total_users, total_doctors=total_doctors,
        total_predictions=total_predictions, total_feedback=total_feedback,
        class_counts=json.dumps(class_counts))

@app.route('/admin/doctor/add', methods=['POST'])
@admin_required
def add_doctor():
    name           = request.form.get('name', '').strip()
    specialization = request.form.get('specialization', '').strip()
    phone          = request.form.get('phone', '').strip()
    email          = request.form.get('email', '').strip()
    hospital       = request.form.get('hospital', '').strip()
    if not name or not specialization or not phone:
        flash('Name, specialization and phone are required.', 'danger')
        return redirect(url_for('admin_dashboard'))
    db.session.add(Doctor(name=name, specialization=specialization,
                          phone=phone, email=email, hospital=hospital))
    db.session.commit()
    flash(f'Dr. {name} added successfully.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/doctor/update/<int:doctor_id>', methods=['POST'])
@admin_required
def update_doctor(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    doctor.name           = request.form.get('name', doctor.name).strip()
    doctor.specialization = request.form.get('specialization', doctor.specialization).strip()
    doctor.phone          = request.form.get('phone', doctor.phone).strip()
    doctor.email          = request.form.get('email', doctor.email).strip()
    doctor.hospital       = request.form.get('hospital', doctor.hospital).strip()
    doctor.available      = request.form.get('available') == 'on'
    db.session.commit()
    flash('Doctor updated successfully.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/doctor/delete/<int:doctor_id>', methods=['POST'])
@admin_required
def delete_doctor(doctor_id):
    db.session.delete(Doctor.query.get_or_404(doctor_id))
    db.session.commit()
    flash('Doctor removed.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/user/delete/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    if user_id == current_user.id:
        flash('Cannot delete your own account.', 'danger')
        return redirect(url_for('admin_dashboard'))
    user = User.query.get_or_404(user_id)
    Prediction.query.filter_by(user_id=user.id).delete()
    Feedback.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.username} deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/feedback/delete/<int:fb_id>', methods=['POST'])
@admin_required
def delete_feedback(fb_id):
    db.session.delete(Feedback.query.get_or_404(fb_id))
    db.session.commit()
    flash('Feedback deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/prediction/delete/<int:pred_id>', methods=['POST'])
@admin_required
def delete_prediction(pred_id):
    db.session.delete(Prediction.query.get_or_404(pred_id))
    db.session.commit()
    flash('Prediction record deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

# ── DB Init ───────────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    # Create default admin
    if not User.query.filter_by(username='admin').first():
        db.session.add(User(
            username='admin', email='admin@skinai.com',
            password=generate_password_hash('admin'), is_admin=True
        ))
        db.session.commit()
    # Seed sample doctors if none exist
    if Doctor.query.count() == 0:
        sample_doctors = [
            Doctor(name='Sarah Mitchell',  specialization='Dermatology',          phone='+1 (555) 201-4892', email='s.mitchell@skincare.com',  hospital='City Dermatology Center'),
            Doctor(name='James Okonkwo',   specialization='Dermatologic Oncology', phone='+1 (555) 334-7710', email='j.okonkwo@medcenter.org',  hospital='Metro Cancer & Skin Institute'),
            Doctor(name='Priya Sharma',    specialization='Cosmetic Dermatology',  phone='+1 (555) 478-1293', email='p.sharma@cosmederm.com',    hospital='Aesthetic Skin Clinic'),
            Doctor(name='Carlos Rivera',   specialization='Dermatopathology',      phone='+1 (555) 562-0088', email='c.rivera@pathlab.net',      hospital='Regional Pathology Lab'),
            Doctor(name='Emily Chen',      specialization='Pediatric Dermatology', phone='+1 (555) 693-5521', email='e.chen@childrenskin.org',   hospital="Children's Skin Health Center"),
            Doctor(name='David Thompson',  specialization='Mohs Surgery',          phone='+1 (555) 771-3340', email='d.thompson@mohsurgery.com', hospital='Advanced Skin Surgery Institute'),
        ]
        db.session.bulk_save_objects(sample_doctors)
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
