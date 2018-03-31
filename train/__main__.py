from dataset import EmopyTalkingDetectionDataset
from models import EmopyTalkingDetectionModel,emopyFeatureExtractedModel
from train import train_model
import os

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # emopy_model = EmopyTalkingDetectionModel(,"/home/mtk/iCog/github.com/Emopy-Models/models/ava.h5")

    model = emopyFeatureExtractedModel(30)
    
    
    dataset = EmopyTalkingDetectionDataset("/dataset/yawn/images3","/dataset/yawn/faces","/home/mtk/iCog/github.com/Emopy-Models/models/ava.json","/home/mtk/iCog/github.com/Emopy-Models/models/ava.h5")
    dataset.load_dataset()
    train_model(dataset,model)

if __name__ == '__main__':
    main()