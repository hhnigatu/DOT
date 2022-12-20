from PIL import Image
import os
import pandas as pd
import torch
import tqdm
from transformers import (
    AdamW,
    LayoutLMv2FeatureExtractor,
    LayoutLMv2TokenizerFast,
    LayoutLMv2Processor,
    LayoutLMv2ForSequenceClassification,
)


def get_images_and_labels(path):
    labels = []
    images = []
    for label_folder, _, file_names in os.walk(path):
        if label_folder != path:
            label = label_folder.split("/")[7]
            for _, _, image_names in os.walk(label_folder):
                relative_image_names = []
                for image_file in image_names:
                    relative_image_names.append(path + "/" + label + "/" + image_file)
                images.extend(relative_image_names)
                labels.extend([label] * len(relative_image_names))
    data = pd.DataFrame.from_dict({"image_path": images, "label": labels})
    return data


def validate(model, valdataloader, validate_data):
    num_correct = 0
    model.eval()
    running_loss = 0.0
    for batch in valdataloader:
        outputs = model(**batch)

        predictions = outputs.logits.argmax(-1)

        num_correct += (predictions == batch["labels"]).float().sum()
        running_loss += outputs.loss.item()

    accuracy = 100 * num_correct / len(validate_data)
    print("Validation accuracy:", accuracy.item())

    return accuracy.item()


def train(
    model, device, optimizer, epochs, train_data, labels, traindataloader, valdataloader
):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    training_loss = []
    validation_accuracy = []
    training_accuracy = []
    validation_loss = []
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Linear(2304, len(labels))

    model.to(torch.device(device))
    assert model.training

    global_step = 0

    for epoch in range(epochs):
        print("Epoch:", epoch)
        running_loss = 0.0
        correct = 0
        for batch in tqdm.tqdm(traindataloader):
            # print(batch.keys())
            # forward pass
            # batch['input_ids'].shape
            outputs = model(**batch)
            loss = outputs.loss
            # print(outputs)
            running_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == batch["labels"]).float().sum()

            # backward pass to get the gradients
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        print("Loss:", running_loss / batch["input_ids"].shape[0])
        accuracy = 100 * correct / len(train_data)
        print("Training accuracy:", accuracy.item())
        validation_accuracy.append(validate(model, valdataloader))


def preprocess_data(
    batch,
    processor,
    column="rep_image",
    labeled=False,
    labels=["image", "other", "narrative", "form", "handwriting", "interview"],
):
    """
    param:
    batch: a batch of paths

    returns
    encoded_inputs: the encoding of the images after running layoutlmv2 processor.

    """
    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}
    images = [Image.open(path).convert("RGB") for path in batch[column]]
    # print(batch)
    with torch.no_grad():
        encoded_inputs = processor(images, padding="max_length", truncation=True)
        if labeled:
            # print('createing labels')
            encoded_inputs["labels"] = [label2id[label] for label in batch["label"]]
    return encoded_inputs
