# dataset directory

### All dataset root : ~/workspace/ocr/datasets

- AIHub : AIHub 출처 데이터

    ㄴ 13. 한국어글자체 

    ㄴ 다양한 형태의 한글 문자 OCR

    ㄴ aihub_crop (luna가 crop해준 데이터)

- ICDAR_2017_2019_Korean : ICDAR2017, ICDAR2019 한국어 데이터 

    ㄴ Training

    ㄴ Validation

    ㄴ crop_Validation : Validation set crop

- picturetransfer : 카카오뱅크 복붙이체 데이터 (positive & negative)


### Datasets for Recognizer model training : ~/workspace/ocr/trainer/all_data

(출처 : ~/workspace/ocr/datasets/ICDAR_2017_2019_Korean/crop_Validation/crop_Validation)

- Training dataset : ./train_data 
    
    *cropped_img_{}_{}.jpg*

    *label.csv*

- validation dataset : ./val_data

    *cropped_img_{}_{}.jpg*

    *label.csv*