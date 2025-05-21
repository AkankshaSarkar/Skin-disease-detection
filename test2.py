import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import cv2
from streamlit_option_menu import option_menu

# Ignore deprecation warning preview for use_column_width
# Replaced with use_container_width as recommended

# Dictionary for class names
class_names = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Nevus',
    6: 'Vascular lesions'
}

# Disease info in both English and Hindi
disease_info = {
    0: {
        "en": {
            "cause": "Long-term exposure to UV radiation from the sun.",
            "treatment": "Cryotherapy, topical creams, or laser therapy."
        },
        "hi": {
            "cause": "सूरज की UV किरणों के लंबे समय तक संपर्क में रहने से।",
            "treatment": "क्रायोथेरेपी, टॉपिकल क्रीम या लेजर थेरेपी।"
        }
    },
    1: {
        "en": {
            "cause": "DNA damage from UV radiation causing uncontrolled cell growth.",
            "treatment": "Surgical excision, Mohs surgery, or radiation therapy."
        },
        "hi": {
            "cause": "UV किरणों से डीएनए को नुकसान, जिससे अनियंत्रित कोशिका वृद्धि होती है।",
            "treatment": "शल्य चिकित्सा, मोह्स सर्जरी या विकिरण थेरेपी।"
        }
    },
    2: {
        "en": {
            "cause": "Non-cancerous skin growths, often from sun exposure or aging.",
            "treatment": "Cryotherapy or shave excision if removal is needed."
        },
        "hi": {
            "cause": "गैर-कैंसरयुक्त त्वचा वृद्धि, आमतौर पर सूर्य के संपर्क या उम्र के कारण।",
            "treatment": "यदि आवश्यक हो तो क्रायोथेरेपी या शेव एक्ससीजन।"
        }
    },
    3: {
        "en": {
            "cause": "Overgrowth of fibroblasts in the skin after minor trauma.",
            "treatment": "Usually no treatment needed; surgical removal if symptomatic."
        },
        "hi": {
            "cause": "हल्की चोट के बाद त्वचा में फाइब्रोब्लास्ट्स की अधिक वृद्धि।",
            "treatment": "आमतौर पर इलाज की आवश्यकता नहीं होती; लक्षण होने पर शल्य चिकित्सा।"
        }
    },
    4: {
        "en": {
            "cause": "Genetic mutations from UV exposure or hereditary factors.",
            "treatment": "Surgical removal, immunotherapy, targeted therapy, or chemotherapy."
        },
        "hi": {
            "cause": "UV संपर्क या आनुवंशिक कारकों से उत्पन्न आनुवंशिक परिवर्तन।",
            "treatment": "सर्जरी, इम्यूनोथेरेपी, टार्गेटेड थेरेपी या कीमोथेरेपी।"
        }
    },
    5: {
        "en": {
            "cause": "Clusters of pigmented skin cells, often genetic or from sun exposure.",
            "treatment": "No treatment unless changes appear; biopsy if needed."
        },
        "hi": {
            "cause": "रंजित त्वचा कोशिकाओं के समूह, अक्सर आनुवंशिक या सूर्य के संपर्क से।",
            "treatment": "जब तक बदलाव न दिखे तब तक इलाज आवश्यक नहीं; आवश्यकता पड़ने पर बायोप्सी।"
        }
    },
    6: {
        "en": {
            "cause": "Abnormal growth of blood vessels under the skin.",
            "treatment": "Laser treatment or corticosteroid therapy."
        },
        "hi": {
            "cause": "त्वचा के नीचे रक्त वाहिकाओं की असामान्य वृद्धि।",
            "treatment": "लेजर थेरेपी या कॉर्टिकोस्टेरॉइड उपचार।"
        }
    }
}

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('newmodel.h5')
    return model

model = load_model()

def predict_image(pil_img, model):
    pil_img = pil_img.resize((100, 75))
    img_array = tf.keras.preprocessing.image.img_to_array(pil_img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return predicted_class

# Language options
languages = {
    "English": {
        "code": "en",
        "title": "Skin Cancer Classification",
        "upload_prompt": "Upload an image for classification",
        "choose_image": "Choose an image...",
        "prediction_result": "### Prediction:",
        "class_label": "Class: ",
        "cause": "Cause: ",
        "treatment": "Treatment: ",
    },
    "Hindi": {
        "code": "hi",
        "title": "त्वचा कैंसर वर्गीकरण",
        "upload_prompt": "वर्गीकरण के लिए एक छवि अपलोड करें",
        "choose_image": "एक छवि चुनें...",
        "prediction_result": "### पूर्वानुमान:",
        "class_label": "वर्ग:",
        "cause": "कारण:",
        "treatment": "उपचार:",
    }
}

# Sidebar for language selection and input method
with st.sidebar:
    selected_language = st.selectbox("Select Language", options=list(languages.keys()), index=0)
    selected_input = option_menu("Input Source", ["Upload Image", "Camera"],
                           icons=['cloud-upload', 'camera'], menu_icon="cast", default_index=0)

lang = languages[selected_language]
lang_code = lang["code"]

# Streamlit UI
st.title(lang["title"])
st.write(lang["upload_prompt"])

def display_results(image, predicted_class):
    st.image(image, caption="Processed Image", use_container_width=True)
    st.write(lang["prediction_result"])
    st.write(f"{lang['class_label']} {class_names[predicted_class]}")
    st.write(f"{lang['cause']} {disease_info[predicted_class][lang_code]['cause']}")
    st.write(f"{lang['treatment']} {disease_info[predicted_class][lang_code]['treatment']}")

# Upload Image option
if selected_input == "Upload Image":
    uploaded_file = st.file_uploader(lang["choose_image"], type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        predicted_class = predict_image(pil_img=image, model=model)
        display_results(image, predicted_class)


# Camera option
elif selected_input == "Camera":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        predicted_class = predict_image(image, model)
        display_results(image, predicted_class)
