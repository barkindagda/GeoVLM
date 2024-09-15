import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from VIGOR import VIGOR  
from model.model import ReRankingModule, ImageEncoder, Configuration, text_encoder
from sentence_transformers import SentenceTransformer
import os
import pickle
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use GPU with ID 2

def extract_embeddings(data_loader, image_encoder, text_embedder, device):
    image_embeddings = []
    text_embeddings = []
    labels = []

    for images, captions, label in data_loader:
        images = images.to(device)
        with torch.no_grad():
            img_emb = image_encoder(images).cpu().numpy()
            txt_emb = text_embedder.encode(captions, convert_to_tensor=True).cpu().numpy()

        image_embeddings.append(img_emb)
        text_embeddings.append(txt_emb)
        labels.append(label.cpu().numpy())

    return np.vstack(image_embeddings), np.vstack(text_embeddings), np.hstack(labels)

def get_all_reference_images_in_batches(image_encoder, dataset, batch_size, device):
    all_ref_images = []
    for i in range(0, len(dataset.test_sat_list), batch_size):
        batch_images = []
        for img_path in dataset.test_sat_list[i:i+batch_size]:
            img = Image.open(img_path).convert('RGB')
            img = dataset.transform_reference(img)
            batch_images.append(img)
        batch_images = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_embs = image_encoder(batch_images).cpu().numpy()
        all_ref_images.append(batch_embs)
    return np.vstack(all_ref_images)

def main():
    # Initialize configuration
    config = Configuration()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_encoder = ImageEncoder(config)
    image_encoder = nn.DataParallel(image_encoder).to(device)  # Use DataParallel for image encoder
    
    # Sentence transformers can be used for the users who don't have access to GPT API key 
    #text_embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Use this if you have access to GPT API
     text_embedder= text_encoder
    
    # Initialize dataset and dataloader
    test_dataset_query = VIGOR(mode='test_query', root='PATH TO VIGOR data',
                               captions_root='PATH TO  VIGOR captions', args=config)
    test_loader_query = DataLoader(test_dataset_query, batch_size=config.batch_size, shuffle=False,
                                   num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn)

    test_dataset_reference = VIGOR(mode='test_reference', root='PATH TO VIGOR data',
                                   captions_root='PATH TO VIGOR captions', args=config)
    test_loader_reference = DataLoader(test_dataset_reference, batch_size=config.batch_size, shuffle=False,
                                       num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Extract embeddings
    query_img_emb, query_txt_emb, query_labels = extract_embeddings(test_loader_query, image_encoder, text_embedder, device)

    # Extract reference embeddings
    ref_img_emb = get_all_reference_images_in_batches(image_encoder, test_dataset_reference, batch_size=6, device=device)
    ref_captions = test_loader_reference.dataset.get_reference_captions(torch.arange(len(ref_img_emb)))
    ref_txt_emb = text_embedder.encode(ref_captions, convert_to_tensor=True).cpu().numpy()

    print('Embeddings ready for use')

    # Initialize reranking model
    # Change the text_dimension to 384 if using Sentence transfromer
    model = ReRankingModule(image_dim=1024, text_dim=1536, common_dim=512)
    model = nn.DataParallel(model).to(device)  # Use DataParallel for the reranking module

    # Load the pretrained reranking model checkpoint
    model.load_state_dict(torch.load('rerank_model_cross_area.pth'))
    model.eval()

    # Perform validation
    top1_accuracy, top5_accuracy, top10_accuracy = validate(query_img_emb, query_txt_emb, query_labels, ref_img_emb, ref_txt_emb, model, device)
    print(f'Test Top-1 Accuracy: {top1_accuracy:.2f}%')
    print(f'Test Top-5 Accuracy: {top5_accuracy:.2f}%')
    print(f'Test Top-10 Accuracy: {top10_accuracy:.2f}%')


def validate(query_img_emb, query_txt_emb, query_labels, ref_img_emb, ref_txt_emb, model, device):
    total_queries = len(query_img_emb)
    correct_reranks_top1 = 0
    correct_reranks_top5 = 0
    correct_reranks_top10 = 0

    for idx in tqdm(range(total_queries), desc="Validating"):
        query_img = torch.tensor(query_img_emb[idx]).unsqueeze(0).to(device)
        query_text = torch.tensor(query_txt_emb[idx]).unsqueeze(0).to(device)
        query_label = query_labels[idx]

        # Compute similarities using numpy
        similarities = np.dot(query_img_emb[idx], ref_img_emb.T)
        top10_indices = np.argsort(similarities)[-10:][::-1]

        scores = []
        for ref_idx in top10_indices:
            ref_img = torch.tensor(ref_img_emb[ref_idx]).unsqueeze(0).to(device)
            ref_text = torch.tensor(ref_txt_emb[ref_idx]).unsqueeze(0).to(device)

            score = model(query_img, query_text, ref_img, ref_text)
            scores.append(score)

        scores = torch.stack(scores).squeeze()

        # Check if reranking is correct for Top-1
        if top10_indices[torch.argmax(scores)] == query_label:
            correct_reranks_top1 += 1

        # Check if reranking is correct for Top-5
        if query_label in top10_indices[:5]:
            correct_reranks_top5 += 1

        # Check if reranking is correct for Top-10
        if query_label in top10_indices:
            correct_reranks_top10 += 1

    top1_accuracy = correct_reranks_top1 / total_queries * 100.0
    top5_accuracy = correct_reranks_top5 / total_queries * 100.0
    top10_accuracy = correct_reranks_top10 / total_queries * 100.0

    return top1_accuracy, top5_accuracy, top10_accuracy


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    main()
