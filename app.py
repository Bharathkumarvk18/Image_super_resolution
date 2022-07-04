import os
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ISR.models import RRDN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from SSIM_PIL import compare_ssim
from math import log10, sqrt


@st.cache(suppress_st_warning=True)
def predict(img):
    lr_img = np.array(img)
    model = RRDN(weights='gans')
    sr_img = model.predict(np.array(lr_img))
    return (Image.fromarray(sr_img))

@st.cache(suppress_st_warning=True)
def feature_detection(image):
      
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()    
    keypoints = sift.detect(gray, None)
     
    st.write("Number of keypoints Detected are: ",len(keypoints))
    image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(image, use_column_width=True,clamp = True)
    
    
    st.write("FAST")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(gray, None)
    st.write("Number of keypoints Detected: ",len(keypoints))
    image_  = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    st.image(image_, use_column_width=True,clamp = True)

@st.cache(suppress_st_warning=True)
def detect_objects(our_image):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    col1, col2 = st.beta_columns(2)

    col1.subheader("Original Image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(our_image)
    col1.pyplot(use_column_width=True)

    # YOLO ALGORITHM
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0,255,size=(len(classes), 3))   


    # LOAD THE IMAGE
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    height,width,channels = img.shape


    # DETECTING OBJECTS (CONVERTING INTO BLOB)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)   #(image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes =[]

    # SHOWING INFORMATION CONTAINED IN 'outs' VARIABLE ON THE SCREEN
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)  
            confidence = scores[class_id] 
            if confidence > 0.5:   
                # OBJECT DETECTED
                #Get the coordinates of object: center,width,height  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  #width is the original width of image
                h = int(detection[3] * height) #height is the original height of the image

                # RECTANGLE COORDINATES
                x = int(center_x - w /2)   #Top-Left x
                y = int(center_y - h/2)   #Top-left y

                #To organize the objects in array so that we can extract them later
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    score_threshold = st.sidebar.slider("Confidence Threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)      
    print(indexes)

    font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            #To get the name of object
            label = str.upper((classes[class_ids[i]]))   
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)     
            items.append(label)


    st.text("")
    col2.subheader("Object-Detected Image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    col2.pyplot(use_column_width=True)

    if len(indexes)>1:
        st.success("Found {} Objects - {}".format(len(indexes),[item for item in set(items)]))
    else:
        st.success("Found {} Object - {}".format(len(indexes),[item for item in set(items)]))

def manual_resize(img):
    im = np.array(img)
    h, w ,c= im.shape
    
    # if(h>250 or w>250):
    #     result=cv2.resize(im,(1000,1000))
    #     return Image.fromarray(result)
    # else:
    result=cv2.resize(im,(w*4,h*4))
    return Image.fromarray(result)


def PSNR(original,compressed ):
    numpy_original=np.array(original)
    numpy_compressed=np.array(compressed)
    cv_original=cv2.cvtColor(numpy_original, cv2.COLOR_RGB2BGR)
    cv_compressed=cv2.cvtColor(numpy_compressed, cv2.COLOR_RGB2BGR)

    mse = np.mean((cv_original - cv_compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr , mse



def object_main():
    """OBJECT DETECTION APP"""

    st.title("SRGANs , Object Dectection(Yolo) , Feature Detection")
    st.write("A model with variety of Machine learning features that can applied on images to obtain enhanced image and object from it")

    choice = st.sidebar.selectbox("Select an option", ("super resolution", "object detection","feature detection"))
    st.write()

    if choice == "object detection":
        st.title("Object detection with YOLO ")
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.subheader("Upload a image which you want to detect objects")
        image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)  
            detect_objects(our_image)

    elif choice == "super resolution":
        #our_image = Image.open("images/person.jpg")
        #detect_objects(our_image)
        st.title("Super Resolution using GANs (SRGANs) ")
        st.subheader("Upload a image which you want to upscale")   
        st.spinner("Testing spinner")
        uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.')
            st.write("")
            if st.button('Upscale'):
                st.write("Resizing...") 
                
                manual=manual_resize(image)
                st.write("Upscaling...")               
                pred = predict(image)

                # value=PSNR(image,manual)
                psnr_value,mse1=PSNR(manual,pred)
                ssim_value=compare_ssim(manual,pred)
                

                # images=[manual,pred]
                # captions=["Manually Resized","Upscaled Image"]
                
                st.image(manual,caption="Manually resized",use_column_width=True)
                st.image(pred, caption='Upscaled Image ', use_column_width=True)
                
                
                  
                st.write("The super resoluted image has decreased  noise by a scale of :",psnr_value,"points")
                st.write("Structural integrity of above images has been improved by a scale of :",(ssim_value)*100 ,"%")
                st.write("Mean Squared error (Comuptes error scale and factor by which details are lost):",mse1)
                
    elif choice == "feature detection":
            st.title("Feature Detection ")
            st.subheader("Upload a image to detect features")
            image1 =st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))
            if image1 is not None:
                image1 = Image.open(image1)
                st.image(image1, caption='Input', use_column_width=True)
                img_array = np.array(image1)
                feature_detection(img_array)
               

if __name__ == '__main__':
    object_main()



