import numpy as np 
import pandas as pd 
import os 
import dlib 
import random
import cv2
import json
def get_dlib_points(image,predictor):
    """
    Get dlib facial key points of face
    Parameters
    ----------
    image : numpy.ndarray
        face image.
    Returns
    -------
    numpy.ndarray
        68 facial key points
    """
    face = dlib.rectangle(0,0,image.shape[1]-1,image.shape[0]-1)
    img = image.reshape(48,48)
    shapes = predictor(img,face)
    parts = shapes.parts()
    output = np.zeros((68,2))
    for i,point in enumerate(parts):
        output[i]=[point.x,point.y]
    output = np.array(output).reshape((1,68,2))
    return output
def to_dlib_points(images,predictor):
    """
    Get dlib facial key points of faces
    Parameters
    ----------
    images : numpy.ndarray
        faces image.
    Returns
    -------
    numpy.ndarray
        68 facial key points for each faces
    """
    output = np.zeros((len(images),1,68,2))
    centroids = np.zeros((len(images),2))
    for i in range(len(images)):
        dlib_points = get_dlib_points(images[i],predictor)[0]
        centroid = np.mean(dlib_points,axis=0)
        centroids[i] = centroid
        output[i][0] = dlib_points
    return output,centroids
        
def get_distances_angles(all_dlib_points,centroids):
    """
    Get the distances for each dlib facial key points in face from centroid of the points and
    angles between the dlib points vector and centroid vector.
    Parameters
    ----------
    all_dlib_points : numpy.ndarray
        dlib facial key points for each face.
    centroid :
        centroid of dlib facial key point for each face
    Returns
    -------
    numpy.ndarray , numpy.ndarray
        Dlib landmarks distances and angles with respect to respective centroid.
    """
    all_distances = np.zeros((len(all_dlib_points),1,68,1))
    all_angles = np.zeros((len(all_dlib_points),1,68,1))
    for i in range(len(all_dlib_points)):
        dists = np.linalg.norm(centroids[i]-all_dlib_points[i][0],axis=1)
        angles = get_angles(all_dlib_points[i][0],centroids[i])
        all_distances[i][0] = dists.reshape(1,68,1)
        all_angles[i][0] = angles.reshape(1,68,1)
    return all_distances,all_angles
def angle_between(p1, p2):
    """
    Get clockwise angle between two vectors
    Parameters
    ----------
    p1 : numpy.ndarray
        first vector.
    p2 : numpy.ndarray
        second vector.
    Returns
    -------
    float
        angle in radiuns
    """
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 - ang2) % (2 * np.pi)
def get_angles(dlib_points,centroid):
    """
    Get clockwise angles between dlib landmarks of face and centroid of landmarks.
    Parameters
    ----------
    dlib_points : numpy.ndarray
        dlib landmarks of face.
    centroid : numpy.ndarray
        centroid of dlib landrmask.
    Returns
    -------
    numpy.ndarray
        dlib points clockwise angles in radiuns with respect to centroid vector
    """
    output = np.zeros((68))
    for i in range(68):
        angle = angle_between(dlib_points[i],centroid)
        output[i] = angle
    return output




class EmopyTalkingDetectionDataset(object):
    def __init__(self,dataset_dir,bounding_boxes_path, *args):
        super(EmopyTalkingDetectionDataset, self).__init__(*args)
        self.dataset_dir  = dataset_dir
        self.image_shape = (48,48,1)
        self.bounding_boxes_path = bounding_boxes_path
        self.dataset_loaded = False
        self.max_sequence_length = 30
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.max_batch_images = 100
    def load_dataset(self):
        train_seq = pd.read_pickle(os.path.join(self.dataset_dir,"train_all.pkl"))
        test_seq = pd.read_pickle(os.path.join(self.dataset_dir,"test_all.pkl"))
        val_seq = pd.read_pickle(os.path.join(self.dataset_dir,"validation_all.pkl"))
        self.train = train_seq["sequence"].as_matrix()
        self.test = test_seq["sequence"].as_matrix()
        self.validation = val_seq["sequence"].as_matrix()
        self.dataset_loaded = True
    def pad_sequence(self,array):
        if len(array)<self.max_sequence_length:
            raise Exception("Sequence length less than max_sequence")
        if len(array)%self.max_sequence_length==0:
            return array
        output = array[:(len(array)//self.max_sequence_length)*self.max_sequence_length]
        output.append(array[len(array)-self.max_sequence_length:])
        return output

    def load_sequence_dataset_helper(self,sequence,img_files,bboxes):
        faces = np.zeros((self.max_sequence_length,48,48,1))
        dpts = np.zeros((self.max_sequence_length,1,68,2))
        dpts_dists = np.zeros((self.max_sequence_length,1,68,1))
        dpts_angles = np.zeros((self.max_sequence_length,1,68,1))

        for i in range(len(img_files)):
            img = cv2.imread(os.path.join(self.dataset_dir,sequence,img_files[i]))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            face = bboxes[img_files[i]]
            face_image = img[
                max(0,int(face[1])):min(img.shape[0],int(face[3])),
                max(0,int(face[0])):min(img.shape[1],int(face[2]))
            ]
            face_image = cv2.resize(face_image,(48,48))
            face_image = face_image.reshape(-1, 48, 48, 1)
            dlibpoints,centroids = to_dlib_points(face_image,self.predictor)
            dists,angles = get_distances_angles(dlibpoints,centroids)

            face_image = face_image.astype('float32')/255
            dlibpoints = dlibpoints.astype(float)/50;
            dists = dists.astype(float)/50;
            angles = angles.astype(float)/(2*np.pi);

            faces[i] = face_image
            dpts[i] = dlibpoints
            dpts_dists[i] = dists
            dpts_angles[i] = angles 
        return faces,dpts,dpts_dists,dpts_angles

    def __get_is_talking(self,sequence_name):
        action = sequence_name.split("-")
        if action[2].lower().count("talking")>0:
            return 1
        else:
            return 0
    def __get_bounding_boxes(self,sequence_name):
        
        with open(os.path.join(self.bounding_boxes_path,sequence_name+".json")) as bb_file:
            return json.load(bb_file)


    def load_sequence_dataset(self,sequence,img_files):

        num_sequences = int(np.ceil(len(img_files)/float(self.max_sequence_length)))

        faces = np.zeros((num_sequences,self.max_sequence_length,48,48,1))
        dlib_points = np.zeros((num_sequences,self.max_sequence_length,1,68,2))
        dpts_dists = np.zeros((num_sequences,self.max_sequence_length,1,68,1))
        dpts_angles = np.zeros((num_sequences,self.max_sequence_length,1,68,1))
        is_talking = np.zeros((num_sequences,2))

        bboxes = self.__get_bounding_boxes(sequence)
        current_sequence = 0
        for i in range(num_sequences):
            fs,dpts,dists,angles = self.load_sequence_dataset_helper(sequence,img_files[i*num_sequences:(i+1)*num_sequences],bboxes)

            faces[i] = fs
            dlib_points[i] = dpts
            dpts_dists[i] = dists
            dpts_angles[i] = angles
            is_talking[i] = np.eye(2)[self.__get_is_talking(sequence)]
            
        return faces,dlib_points,dpts_dists,dpts_angles,is_talking



        

    def generate_indexes(self,length):
        output = range(length)
        random.shuffle(output)
        return output
    def train_generator(self):
        
        while True:
            indexes = self.generate_indexes(len(self.train))
            for i in range(len(indexes)):
                index = indexes[i]
                img_files = os.listdir(os.path.join(self.dataset_dir,self.train[index]))
                img_files.sort()
                for i in range(0,len(img_files)-self.max_batch_images,self.max_batch_images):
                    current_images = img_files[i:i+self.max_batch_images]
                    faces,dpts,dpts_dists,dpts_angles,y = self.load_sequence_dataset(self.train[index],current_images)
                    # y = np.eye(2)[y]
                    yield [faces,dpts,dpts_dists,dpts_angles],y
    def test_generator(self):
        
        while True:
            indexes = self.generate_indexes(len(self.test))
            for i in range(len(indexes)):
                index = indexes[i]
                img_files = os.listdir(os.path.join(self.dataset_dir,self.test[index]))
                img_files.sort()
                for i in range(0,len(img_files)-self.max_batch_images,self.max_batch_images):
                    current_images = img_files[i:i+self.max_batch_images]
                    faces,dpts,dpts_dists,dpts_angles,y = self.load_sequence_dataset(self.test[index],current_images)
                    # y = np.eye(2)[y]
                    yield [faces,dpts,dpts_dists,dpts_angles],y
    def validation_generator(self):
        
        while True:
            indexes = self.generate_indexes(len(self.validation))
            for i in range(len(indexes)):
                index = indexes[i]
                img_files = os.listdir(os.path.join(self.dataset_dir,self.validation[index]))
                img_files.sort()
                for i in range(0,len(img_files)-self.max_batch_images,self.max_batch_images):
                    current_images = img_files[i:i+self.max_batch_images]
                    faces,dpts,dpts_dists,dpts_angles,y = self.load_sequence_dataset(self.validation[index],current_images)
                    # y = np.eye(2)[y]
                    yield [faces,dpts,dpts_dists,dpts_angles],y