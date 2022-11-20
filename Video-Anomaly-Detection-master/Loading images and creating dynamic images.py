### Loading images of activity of daily living

images = []
res_img = []
label = []
res_label = []
y = []
res_y = []
Images = []

adl = os.listdir(r"C:\Users\Arun\Documents\Video Anomaly Detection\Dataset\ADL camera 0 RGB Data")
filepath = r"C:\Users\Arun\Documents\Video Anomaly Detection\Dataset\ADL camera 0 RGB Data"

for i in adl:
    
   k = os.path.join(filepath,i)
   n = os.listdir(k)
   
   z = np.zeros((227, 227, 3))
   cnt = 0
   
   for j in n:
       
       img = image.load_img(os.path.join(k, j), target_size = (227, 227, 3))
       Images.append(img)
       img_arr = image.img_to_array(img)
       images.append(img_arr)
       label.append(0)
       y.append(j)
       
       z += img_arr
       cnt += 1
       
       
   z = z/cnt
   z = z/255
   res_img.append(z)
   res_label.append(0)
   res_y.append(i)       


### Loading images of fall activity
   

fall = os.listdir(r"C:\Users\Arun\Documents\Video Anomaly Detection\Dataset\Fall camera 0 RGB Data")
filepath2 = r"C:\Users\Arun\Documents\Video Anomaly Detection\Dataset\Fall camera 0 RGB Data"

for i in fall:
    
   k = os.path.join(filepath2,i)
   n = os.listdir(k)
   
   z = np.zeros((227, 227, 3))
   cnt = 0
   
   for j in n:
       
       img = image.load_img(os.path.join(k, j), target_size = (227, 227, 3))
       Images.append(img)
       img_arr = image.img_to_array(img)
       images.append(img_arr)
       label.append(1)
       y.append(j)
       
       z += img_arr
       cnt += 1
       
       
   z = z/cnt
   z = z/255
   res_img.append(z)
   res_label.append(1)
   res_y.append(i)   
   
res_img2 = np.array(res_img)
res_label2 = np.array(res_label)
res_label2 = to_categorical(res_label2)
