"""import pickle
file = open("data/ucf101_24/annotations/UCF101v2-GT.pkl",'rb')
object_file = pickle.load(file)"""
import pandas as pd
object = pd.read_pickle(r'/home/kiki/Documents/Cercetare/mmaction2/data/ucf101_24/annotations/UCF101v2-GT.pkl')
# read pickle, extract needed info (e.g train_videos - > annotation_file)
# put values from pickle into config
# in train script, set values for config
# first match values from config that are the same in both ava and ucf 24
# skip values and try to understand what those are - if not found in ucf24
object2 = pd.read_pickle(r'data/ava/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl')

#object2 = pd.read_pickle(r'/home/kiki/Documents/Cercetare/mmaction2/data/ucf101_24/annotations/pyannot.pkl')
#print(object)


print(object2.keys()) #dict_keys(['labels', 'gttubes', 'nframes', 'train_videos', 'test_videos', 'resolution'])
print("OBJECT:", object2['_145Aa_xkuE,0972'])

#print("OBJECT gttubes:", object['gttubes'].keys())
print("OBJECT:", object['gttubes']['WalkingWithDog/v_WalkingWithDog_g05_c02'][23][0][0])




#print(object[['BasketballDunk/v_BasketballDunk_g12_c01'])
#print("DENSE PROPOSALS:", object2['dMH8L7mqCNI,1307']) #dense proposals
#ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c04.avi 1 ->test
#sudo apt-get install -y kdiff3
"""
Dictionary that contains the ground truth tubes for each video. A gttube is dictionary that associates with each index 
of label and a list of tubes. A tube is a numpy array with nframes rows and 5 columns, each col is in format like 
frame_idx x1 y1 x2 y2
"""
#print(object['gttubes']['HorseRiding/v_HorseRiding_g11_c01'])
#print(object2)
#print(object['labels'])