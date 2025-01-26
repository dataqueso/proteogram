"""
Image search approach based on https://github.com/totogot/ImageSimilarity
"""
import torch
import os
import math
import json
from time import time
import glob
from torch.multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import vit_b_16
import torchvision.transforms as transforms

from kmeans_pytorch import kmeans
from PIL import Image, ImageDraw, ImageFont


class Img2Vec:
    """
    Class for embedding dataset of image files into vectors using Pytorch
    standard neural networks.

    Parameters:
    -----------
    model_name: str object specifying neural network architecture to utilise.
        Must align to the naming convention specified on Pytorch documentation:
        https://pytorch.org/vision/main/models.html#classification
        For supported model architectures see self.embed_dict below
    weights: str object specifying the pretrained weights to load into model or
        a file path to a supported ResNet model.
        Only weights supported by Pytorch torchvision library can be accessed.
        Current functionality reverts to DEFAULT weights if no specified.

    See also:
    -----------
    Img2Vec.embed_dataset(): embed passed images as feature vectors
    Img2Vec.save_dataset(): save embedded dataset to file for future loading
    Img2Vec.load_dataset(): load previously embedded dataset of feature vectors
    Img2Vec.similarities(): calculate cosine similarities for the embedding dataset
    Img2Vec.cluster_dataset(): group embedded images into specified n clusters

    Example:
    -----------

    ImgSim = imgsim.Img2Vec('resnet50', weights='DEFAULT')
    ImgSim.embed_dataset('[EXAMPLE PATH TO DIRECTORY OF IMAGES]')

    ImgSim.save_dataset('[OUTPUT PATH FOR SAVING EMBEDDEDINGS]')

    ImgSim.similarities(n=5)

    ImgSim.cluster_dataset(nclusters=6, display=True)
    """

    def __init__(self, model_name_or_path, dataset_dir, embed_file=None, weights="DEFAULT", device=None):
        # dictionary defining the supported NN architectures
        self.embed_dict = {
            "resnet50": self.obtain_children,
            "resnet_ft": self.obtain_children,
            "resnet152": self.obtain_children,
            "resnext50_32x4d": self.obtain_children,
            "resnext101_64x4d": self.obtain_children,
            "convnext_large": self.obtain_children,
            "vgg19": self.obtain_classifier,
            "efficientnet_b0": self.obtain_classifier,
            "vit_b_16": self.obtain_encoder,
        }
        if not device:
            self.device = self.set_device()
        else:
            self.device = torch.device(device)

        # assign class attributes
        self.architecture = self.validate_model(model_name_or_path)
        if self.architecture == "resnet_ft":
            weights = model_name_or_path
        self.vit_model = None
        if self.architecture == "vit_b_16":
            self.vit_model = vit_b_16()
            self.vit_model = self.vit_model.to(self.device).eval()
        self.weights = weights
        self.transform = self.assign_transform(weights)
        self.model = self.initiate_model()
        self.embed = self.assign_layer()
        self.embed_file = embed_file
        self.cosine = nn.CosineSimilarity(dim=1)
        self.dataset = {}
        self.image_clusters = {}
        self.cluster_centers = {}
        self.sim_dict = {}
        self.files = self.validate_source(dataset_dir)

    def validate_model(self, model_name_or_path):
        if os.path.exists(model_name_or_path):
            model_name = "resnet_ft"
        else:
            if model_name_or_path not in self.embed_dict.keys():
                raise ValueError(f"The model {model_name_or_path} is not supported")
            else:
                model_name = model_name_or_path
        return model_name

    def assign_transform(self, weights):
        weights_dict = {
            "resnet50": models.ResNet50_Weights,
            "resnet152": models.ResNet152_Weights,
            "resnext50_32x4d": models.ResNeXt50_32X4D_Weights,
            "resnext101_64x4d": models.ResNeXt101_64X4D_Weights,
            "convnext_large": models.ConvNeXt_Large_Weights,
            "vgg19": models.VGG19_Weights,
            "efficientnet_b0": models.EfficientNet_B0_Weights,
            "vit_b_16": models.ViT_B_16_Weights,
            "resnet_ft": None
        }

        # try load preprocess from torchvision else assign default
        try:
            w = weights_dict[self.architecture]
            weights = getattr(w, weights)
            preprocess = weights.transforms()
        except Exception:
            preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        return preprocess

    def set_device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def initiate_model(self):
        if self.architecture == "resnet_ft":
            model = torch.load(self.weights,
                               weights_only=False,
                               map_location=self.device)
        else:
            m = getattr(
                models, self.architecture
            )  # equ to assigning m as models.resnet50()
            model = m(weights=self.weights)  # equ to models.resnet50(weights=...)
        model.to(self.device)

        return model.eval()

    def assign_layer(self):
        model_embed = self.embed_dict[self.architecture]()

        return model_embed

    def obtain_children(self):
        model_embed = nn.Sequential(*list(self.model.children())[:-1])

        return model_embed

    def obtain_classifier(self):
        self.model.classifier = self.model.classifier[:-1]

        return self.model
    
    def obtain_encoder(self):
        model_embed = nn.Sequential(*list(self.model.children())[:-1])

        # The encoder only
        model_embed = model_embed[1]

        return model_embed

    def directory_to_list(self, dir):
        ext = (".png", ".jpg", ".jpeg")

        d = os.listdir(dir)
        source_list = [os.path.join(dir, f) for f in d if \
                       os.path.splitext(f)[1].lower() in ext]

        return source_list

    def validate_source(self, source):
        # convert source format into standard list of file paths
        if isinstance(source, list):
            source_list = [f for f in source if os.path.isfile(f)]
        elif os.path.isdir(source):
            ext = ["png", "jpg", "jpeg"]
            #source_list = self.directory_to_list(source)
            source_list = glob.glob(os.path.join(source, '**', '*.*'),
                                    recursive=True)
            source_list = [f for f in source_list if \
                           f.split('.')[-1].lower() in ext]
        elif os.path.isfile(source):
            source_list = [source]
        else:
            raise ValueError('"source" expected as file, list or directory.')

        return source_list

    def embed_image_for_mp(self, img_file):
        # load and preprocess image
        img = Image.open(img_file)
        with torch.no_grad():
            img_trans = self.transform(img)

            # store computational graph on GPU if available
            if self.device == "cuda:0":
               img_trans = img_trans.cuda()

            img_trans = img_trans.unsqueeze(0)

        return (img_file, self.embed(img_trans))

    def embed_image(self, img_file):
        # load and preprocess image
        img = Image.open(img_file)
        with torch.no_grad():
            if self.architecture == "vit_b_16":
                img_trans = self.transform(img)
                img_trans = img_trans.unsqueeze(0).to(self.device)
                img_trans = self.vit_model._process_input(img_trans)
                n = img_trans.shape[0]
                batch_class_token = self.vit_model.class_token.expand(n, -1, -1)
                img_trans = torch.cat([batch_class_token, img_trans], dim=1)
                embedded_img = self.embed(img_trans)[:, 0]
            else:
                img_trans = self.transform(img)
                img_trans = img_trans.unsqueeze(0)
                img_trans = img_trans.to(self.device)
                embedded_img = self.embed(img_trans)

        return embedded_img

    def embed_dataset_mp(self, source):
        # convert source to appropriate format
        self.files = self.validate_source(source)
        with torch.no_grad():
            with Pool(1) as pool:
                results = pool.map(self.embed_image_for_mp, self.files)
            for img, embedding in results:
                self.dataset[str(img)] = embedding.clone()#.detach().item()

    def embed_dataset(self):
        # convert source to appropriate format
        with torch.no_grad():
            for img in tqdm(self.files):
                self.dataset[str(img)] = self.embed_image(img)

    def sim_calc(self, image_path1, image_path2):
        embedding1 = self.dataset[image_path1]
        embedding2 = self.dataset[image_path2]
        with torch.no_grad():
            sim = self.cosine(embedding1, embedding2)[0].item()
        return (sim, image_path1, image_path2)

    def embed_and_sim_calc(self, image_paths):
        image_path1, image_path2 = image_paths
        with torch.no_grad():
            _, embedding1 = self.embed_image(image_path1)
            _, embedding2 = self.embed_image(image_path2)
            sim = self.cosine(embedding1, embedding2)[0].item()
        return (sim, image_path1, image_path2)

    def sim_calc_new_embedding(self,
                               image_path_query,
                               image_path_target,
                               embedding_query,
                               embedding_target):
        with torch.no_grad():
            sim = self.cosine(embedding_query, embedding_target)[0].item()
        return (sim, image_path_target)

    def similarities_new_image(self,
                               query_image_path,
                               n=10,
                               save_results_dir=None,
                               save_result_images_dir=None):
        embedding_new = self.embed_image(query_image_path).detach()

        with mp.Pool(os.cpu_count()-2) as pool:
           results = pool.starmap(self.sim_calc_new_embedding,
                               [(query_image_path, image_path_j, embedding_new, embedding_j) for\
                                 (image_path_j, embedding_j) in self.dataset.items()])
        
        scores = {}
        for (sim, image_path_j) in results:
            scores[image_path_j] = sim
        
        scores_n_arr = sorted(scores.items(),
                            key=lambda item: item[1],
                            reverse=True)[:n]

        # # If there's a dir specified in save_result_images_dir, create result image
        self.save_images(query_image_path,
                         save_result_images_dir,
                         scores_n_arr=scores_n_arr)

        return scores_n_arr

    def sim_calc_new_embedding_scratch(self,
                               image_path_query,
                               image_path_target):
        with torch.no_grad():
            embedding_query = self.embed_image(image_path_query)
            embedding_target = self.embed_image(image_path_target)
            sim = self.cosine(embedding_query, embedding_target)[0].item()
        return (image_path_query, image_path_target, sim)

    def similarities_new_image_scratch(self,
                               image_path_query,
                               n=10,
                               save_result_images_dir=None,
                               use_prev_embeddings=False):
        start_sim_calc = time()
        # Check if need to create embeddings for input dataset
        if use_prev_embeddings == False:
            print(f'Creating and saving embedding dataset')
            self.embed_dataset()
            self.save_dataset()
        else:
            print('Using previously embedded dataset')
            self.load_dataset()

        # initiate computation of consine similarity
        cosine = nn.CosineSimilarity(dim=1).to(self.device)
        # Create a dict of similarities (a dict of dict of scores), e.g:
        # this looks like --> sim_dict[image_i] = {dict[image_0], dict[image_1], ...}
        scores = []
        embedding_query = self.embed_image(image_path_query)
        for image_path_i, embedding_i in tqdm(self.dataset.items()):
            sim = cosine(embedding_query, embedding_i)[0].item()
            scores.append((image_path_i, sim))
        # Sort the scores
        scores = sorted(scores,
                        key=lambda item: item[1],
                        reverse=True)
        scores_n_arr = scores[:n]
        sim_calc_time = time() - start_sim_calc

        # If there's a dir specified in save_result_images_dir, create result image
        if save_result_images_dir:
            self.save_images(image_path_query,
                             save_result_images_dir,
                             scores_n_arr=scores_n_arr)

        return scores_n_arr, sim_calc_time

    def similarities(self,
                     n=10,
                     save_results_dir=None,
                     save_result_images_dir=None,
                     use_prev_embeddings=True):
        """
        Function for creating the similarity matrix between embeddings in the dataset
        using cosine similarity.

        Parameters:
        -----------
        n : int
            Specifying the top n most similar images to store (and optionally
            save as images).
        save_results_dir : str
            Directory to save the search results tsv file.
        save_result_images : str
            Directory to store search image results (top K images).
        use_prev_embeddings : bool
            Use self.dataset calculated embeddings (this requires a lot of memory)
            otherwise calculate embeddings with multiprocessing and do not store
            (better memory performance).
        """
        #torch.set_num_threads(1)
        #mp.set_start_method('spawn')
        start_sim_calc = time()

        # Get all pairs of files
        file_pairs = []
        for i in range(len(self.files)):
            for j in range(i+1, len(self.files)):
                file_pairs.append((self.files[i], self.files[j]))
        print(f'done making file_pairs')

        # initiate computation of consine similarity
        cosine = nn.CosineSimilarity(dim=1).to(self.device)
        if use_prev_embeddings:
            print(f'Using previous embeddings')
            # Create a dict of similarities (a dict of dict of scores), e.g:
            # this looks like --> sim_dict[image_i] = {dict[image_0], dict[image_1], ...}
            for image_path_i, embedding_i in tqdm(self.dataset.items()):
                scores = {}
                for image_path_j, embedding_j in self.dataset.items():
                    sim = cosine(embedding_i, embedding_j)[0].item()
                    scores[image_path_j] = sim
                # Sort the scores
                scores = sorted(scores.items(),
                                key=lambda item: item[1],
                                reverse=True)
                scores_arr = []
                for k, (image_path_j, sim) in enumerate(scores):
                    scores_arr.append((image_path_j, sim))
                    if k == n-1:
                        break
                self.sim_dict[image_path_i] = scores_arr
            sim_calc_time = time() - start_sim_calc
        else:
            print(f'# CPUs = {os.cpu_count()}')
            with mp.Pool(os.cpu_count() - 2) as pool:
                results = pool.imap(self.sim_calc_new_embedding_scratch,
                                    tqdm(file_pairs, total=len(file_pairs)))
            # this looks like --> sim_dict[image_i] = {dict[image_0], dict[image_1], ...}
            for (image_path_query, image_path_target, sim) in results:
                if image_path_query in scores:
                    item1 = self.sim_dict[image_path_query] # dict[image_0]
                    if image_path_target not in item1:
                        item1[image_path_target] = sim
                        self.sim_dict[image_path_query] = item1
                else:
                    tmp = {}
                    tmp[image_path_target] = sim # dict[image_0]
                    self.sim_dict[image_path_query] = tmp
            for img_path in self.sim_dict.keys():
                scores = self.sim_dict[img_path]
                # Sort the target images by sim scores
                self.sim_dict[img_path] = sorted(scores.items(),
                                                    key=lambda item: item[1],
                                                    reverse=True)
            # Modify data structure to tuples and capture top k results if n is defined
            # this looks like --> sim_dict[image_i] = [(image_0, score_0), (image_1, score_1), ...]
            for image_path_i, scores_dict in self.sim_dict.items():
                scores_arr = []
                for k, (image_path_j, score) in enumerate(scores_dict.items()):
                    scores_arr.append((image_path_j, score))
                    if n:
                        # Reached top k if n is specified so stop
                        if k == n-1:
                            break
                if len(scores_arr) < n:
                    print(f'Something is wrong because len of scores array = {len(scores_arr)}')
                self.sim_dict[image_path_i] = scores_arr

        sim_calc_time = time() - start_sim_calc

        # If there's a dir specified in save_result_images_dir, create result images
        if save_result_images_dir:
            for image_path in self.sim_dict.keys():
                self.save_images(image_path, save_result_images_dir)

        # Save the search results and scores as json file
        if save_results_dir:
            with open(os.path.join(save_results_dir, 'search_similarity_report.json'), 'w') as fout:
                json.dump(self.sim_dict, fout)

        return sim_calc_time

    def save_images(self, query_file, save_dir, scores_n_arr=None):
        """Save similar images from similarity search"""
        images_files = [query_file]
        if not scores_n_arr:
            images_files.extend([target for target, _ in self.sim_dict[query_file]])
            scores = ['']
            scores.extend([f'{score:.2f}' for _, score in self.sim_dict[query_file]])
        else:
            images_files.extend([file for file, _ in scores_n_arr])
            scores = ['']
            scores.extend([f'{score:.2f}' for _, score in scores_n_arr])            
        images = [Image.open(target) for target in images_files]

        max_height = 1000
        total_width = max_height * len(images)
        font_size = max_height // 15
        path = os.path.dirname(__file__)
        font = ImageFont.truetype(os.path.join(path,'fonts','FreeSansBold.ttf'), font_size)

        new_im = Image.new('RGB', (total_width, max_height+70), color='white')

        x_offset = 0
        for i, im in enumerate(images):
            im = im.resize((max_height, max_height), Image.LANCZOS)
            new_im.paste(im, (x_offset,0))
            I1 = ImageDraw.Draw(new_im)
            if i > 0:
                I1.text((x_offset,max_height-5),
                        os.path.basename(images_files[i][:-4]) + f' = {scores[i]}', 
                        fill=(0, 0, 0),
                        font=font)
            else:
                # Don't need score since this is just the image query
                I1.text((x_offset,max_height-5),
                        os.path.basename(images_files[i][:-4]),
                        fill=(0, 0, 0),
                        font=font)
            x_offset+=max_height
            # x_offset += im.size[0]

        out_filename = save_dir + os.sep + \
            '.'.join(os.path.basename(query_file).split('.')[:-1]) + \
            '_top_sims.jpg'
        new_im.save(out_filename)

        return None
    
    def show_images(self, similar, target):
        self.display_img(target, "original")

        for k, v in similar.items():
            self.display_img(k, "similarity:" + str(v))

        return None

    def display_img(self, path, title):
        plt.imshow(Image.open(path))
        plt.axis("off")
        plt.title(title)
        plt.show()

        return

    def save_dataset(self):
        """
        Function to save a previously embedded image dataset to file

        Parameters:
        -----------
        path: str specifying the output folder to save the tensors to
        """

        # convert embeddings to dictionary
        data = {"model": self.architecture, "embeddings": self.dataset}

        torch.save(
            data, self.embed_file
        )  # need to update functionality for naming convention

    def load_dataset(self):
        """
        Function to save a previously embedded image dataset to file

        Parameters:
        -----------
        source: str specifying tensor.pt file to load previous embeddings
        """

        data = torch.load(self.embed_file)

        # assess that embedding nn matches currently initiated nn
        if data["model"] == self.architecture:
            self.dataset = data["embeddings"]
        else:
            raise AttributeError(
                f'NN architecture "{self.architecture}" does not match the '
                + f'"{data["model"]}" model used to generate saved embeddings.'
                + " Re-initiate Img2Vec with correct architecture and reload."
            )

    def plot_list(self, img_list, cluster_num):
        fig, axes = plt.subplots(math.ceil(len(img_list) / 2), 2)
        fig.suptitle(f"Cluster: {str(cluster_num)}")
        [ax.axis("off") for ax in axes.ravel()]

        for img, ax in zip(img_list, axes.ravel()):
            ax.imshow(Image.open(img))

        fig.tight_layout()

        return

    def display_clusters(self):
        for num in self.cluster_centers.keys():
            # print(f'Displaying cluster: {str(cluster_num)}')

            img_list = [k for k, v in self.image_clusters.items() if v == num]
            self.plot_list(img_list, num)

        return

    def cluster_dataset(self, nclusters, dist="euclidean", display=False):
        vecs = torch.stack(list(self.dataset.values())).squeeze()
        imgs = list(self.dataset.keys())
        np.random.seed(100)

        cluster_ids_x, cluster_centers = kmeans(
            X=vecs, num_clusters=nclusters, distance=dist, device=self.device
        )

        # assign clusters to images
        self.image_clusters = dict(zip(imgs, cluster_ids_x.tolist()))

        # store cluster centres
        cluster_num = list(range(0, len(cluster_centers)))
        self.cluster_centers = dict(zip(cluster_num, cluster_centers.tolist()))

        if display:
            self.display_clusters()

        return
