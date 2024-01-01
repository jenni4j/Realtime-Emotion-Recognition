from ai.dive.data.label_reader import LabelReader
from ai.dive.data.image_file_classification import ImageFileClassificationDataset
from transformers import ViTForImageClassification, ViTImageProcessor
import argparse
import os

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train a ViT on dataset')
    parser.add_argument('-d', '--data', required=True, type=str, help='datasets to train/eval model on')
    parser.add_argument('-o', '--output', required=True, type=str, help='output file to write results to')
    parser.add_argument('-m', '--base_model', default="google/vit-base-patch16-224-in21k", type=str, help='The base model to use')
    parser.add_argument('-g', '--gpu', default=False, help='Train on the GPU if supported')
    args = parser.parse_args()

    labels_file = os.path.join(args.data, "labels.txt")
    label_reader = LabelReader(labels_file)
    labels = label_reader.labels()
    print(labels)

    # Same processor as before that is tied to the model
    processor = ViTImageProcessor.from_pretrained(args.base_model)

   # Load the dataset into memory, and convert to a hugging face dataset
    print("Preparing train dataset...")
    train_file = os.path.join(args.data, "train.csv")
    ds = ImageFileClassificationDataset(
        data_dir=args.data,
        file=train_file,
        label_reader=label_reader,
        img_processor=processor,
        num_samples=100
    )
    train_dataset = ds.to_hf_dataset()

    print(train_dataset[0])
    print(train_dataset[0]['pixel_values'].shape)

if __name__ == '__main__':
    main()
