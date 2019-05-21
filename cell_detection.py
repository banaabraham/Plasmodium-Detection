from sklearn.cluster import KMeans
from skimage.feature import greycomatrix, greycoprops
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import models
from sklearn.cluster import MeanShift
from keras.utils import plot_model
from sklearn import metrics

model = models.load_model('cell_class_v3.h5')
plot_model(model, to_file='model.png')


d = "Deteksi Parasit-20190418T085731Z-001\\Deteksi Parasit\\Data\\2.png"
img = cv2.imread(d)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

(winW, winH) = (100, 100)

detected = []
scores = []
for (x, y, window) in sliding_window(img, stepSize=35, windowSize=(winW, winH)):
    if img.shape[0] < 100 or img.shape[1] < 100:
        break
    c = cv2.resize(window,(100,100))/255
    c = c.reshape(1,100,100,3)
    if model.predict(c)>0.9:
        detected.append((x,y,100,100))
        scores.append(model.predict(c))


verticles = []
for i in detected:
    verticles.append((i[0],i[1],i[0]+i[2],i[1]+i[3]))


centroids = []
for (x1,y1,x2,y2) in verticles:
    centroids.append((x1+(x2-x1)/2,y1+(y2-y1)/2))

if len(centroids)>1:
    centroids = np.array(centroids)
    clustering = MeanShift(bandwidth=100).fit(centroids)
    clust_centers = clustering.cluster_centers_
    
    radial = [[] for i in range(len(np.unique(clustering.labels_)))]
    
    #cluster the bounding boxes
    for i in range(len(verticles)):
        radial[clustering.labels_[i]].append(verticles[i])
    
    length = []
    for r in radial:
        r = np.array(r)
        length.append([int((np.max(r[:,2])-np.min(r[:,0])) /  4) , int( (np.max(r[:,3])-np.min(r[:,1])) / 4 )])
    
    verticles = []
    for i,(x,y) in enumerate(clust_centers):
        verticles.append((int(x-length[i][0]),int(y-length[i][1]),int(x+length[i][0]),int(y+length[i][1])))
    
    
    #if bounding boxes grater than 1 compute glcm and cluster with kmeans
    if len(verticles)>1:
        
        features = []
        for (x,y,w,h) in verticles:
            glcm = greycomatrix(img_gray[y:h,x:w], [6], [0], 256, symmetric=True, normed=True)
            features.append([#greycoprops(glcm, 'dissimilarity')[0, 0],
                             #greycoprops(glcm, 'correlation')[0, 0],
                             #greycoprops(glcm, 'contrast')[0, 0],
                             greycoprops(glcm, 'homogeneity')[0, 0],
                             greycoprops(glcm, 'energy')[0, 0]])
          

        
        k, sil = 0, 0
        if len(verticles)>3:
            for i in range(2,4):
                kmeans = KMeans(n_clusters=i, random_state=0).fit(features)
                labels = list(kmeans.labels_)
                sil_score = metrics.silhouette_score(features, labels, metric='euclidean')
                print(sil_score)
                if sil_score>sil:
                    sil = sil_score
                    k = i
            
            if sil<0.5:
                labels = [0 for i in range(len(verticles))]
            else:
                kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
                labels = list(kmeans.labels_)
        else:
            if len(verticles) == 3:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
                labels = list(kmeans.labels_)
                sil_score = metrics.silhouette_score(features, labels, metric='euclidean')
                if sil_score<0.5:
                    labels = [0 for i in range(len(verticles))]
                    
            elif len(verticles) == 2:
                dist = np.linalg.norm(np.array(features[1])-np.array(features[0]))
                if dist<0.07:
                    labels = [0 for i in range(len(verticles))]
                else:
                    labels = [0,1]
            else:
                labels = [0]
        
        choice = 0
        min_hom = 1000
        if len(labels)>2:
            for i in labels:
                if min_hom > kmeans.cluster_centers_[i,1]:
                    min_hom = kmeans.cluster_centers_[i,1]
                    choice = i
        else:
            if features[0][1]>features[1][1]:
                choice = 1
            else:
                choice = 0
        
        filtered_verticles = []
        
        for index in range(len(verticles)):
            if labels[index] == choice:
                filtered_verticles.append(verticles[index])
        
    else:
        filtered_verticles = verticles
    
              
else:
    vericles = []

       
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

for (x, y, w, h) in verticles:
    cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)

for (x, y, w, h) in filtered_verticles:
    cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)

#cv2.imwrite("hasil/"+d.split("\\")[-1],img)
cv2.imwrite("hasil/two.jpg",img)

cv2.imshow("siap",cv2.resize(img,(854,480)))
cv2.waitKey(0)
cv2.destroyAllWindows()


fitur = np.array(features)

#kmeans = KMeans(n_clusters=3, random_state=0).fit(fitur)
#labels = kmeans.labels_


LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'g',
                   2 : 'b'}
label_color = [LABEL_COLOR_MAP[l] for l in labels]
plt.xlabel('homogeneity', fontsize=18)
plt.ylabel('energy', fontsize=16)
plt.scatter(fitur[:,0], fitur[:,1] , c=label_color)

