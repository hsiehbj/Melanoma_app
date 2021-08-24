import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Melanoma prediction')

# Create file uploader 
img_data = st.file_uploader(label='Load image for recognition', type=['png', 'jpg', 'jpeg'])

@st.cache(allow_output_mutation = True)
def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return (model)


model = load_model("melanoma_b0_fine_tune_25layers")

if img_data is not None:
    
    # Display image
    uploaded_img = Image.open(img_data)
    st.image(uploaded_img)
    
    # Load image file to predict and make prediction
    #img_path = f'Users/hsiehbj/{img_data.name}'
    #img = image.load_img(img_path, target_size=(224,224))
    
    test_image = uploaded_img.resize((224,224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    
    pred = model.predict(test_image)
    class_names = ['Benign nevus', 'Melanoma']
    pred_class = class_names[int(tf.round(pred)[0][0])]
    
    # Display prediction
    st.title('Prediction:')
    st.write(pred_class)
    st.title('Risk')
    st.write(str(pred[0][0]))
else:
    print('No image')