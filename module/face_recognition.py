import os

import inspect
import numpy as np
import torch
from PIL import Image

from constance import ALIGN_MODEL, RECOGNITION_MODEL, REC_THRESHOLD
from database.qdrant_db import Database
from module.utils import dotdict, load_hf_model_by_repo_id, pil_to_input

np.bool = np.bool_  # fix bug for mxnet 1.9.1
np.object = np.object_
np.float = np.float_


class FaceRecognition:
    def __init__(self):
        self.thresh = REC_THRESHOLD
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.database = Database()

        # load model
        self.fr_model, self.aligner = self._load_models(RECOGNITION_MODEL, ALIGN_MODEL)

    def _load_models(self, recognition_model_id: str, aligner_id: str):
        fr_model = load_hf_model_by_repo_id(repo_id=recognition_model_id,
                                            save_path=os.path.expanduser(f"~/.cvlface_cache/{recognition_model_id}"),
                                            HF_TOKEN=os.getenv("HF_TOKEN"), ).to(self.device)
        aligner = load_hf_model_by_repo_id(repo_id=aligner_id,
                                           save_path=os.path.expanduser(f"~/.cvlface_cache/{aligner_id}"),
                                           HF_TOKEN=os.getenv("HF_TOKEN"), ).to(self.device)
        return fr_model, aligner

    def embedding_image(self, image: Image.Image):
        image = image.convert("RGB")
        image = pil_to_input(image)

        # align
        aligned_x, _, aligned_ldmks, _, _, _ = self.aligner(image)

        # recognize
        input_signature = inspect.signature(self.fr_model.model.net.forward)
        if input_signature.parameters.get("keypoints") is not None:
            embed = self.fr_model(aligned_x, aligned_ldmks)
        else:
            embed = self.fr_model(aligned_x)
        return embed

    def _compute_similarity(self, embed_1, embed_2):
        # compute cosine similarity
        cossim = torch.nn.functional.cosine_similarity(embed_1, embed_2).item()
        is_same = cossim > self.thresh
        return is_same, cossim

    def inference(self, image: Image.Image):
        query_embed = self.embedding_image(image)
        query_embed_np = query_embed.detach().cpu().numpy()[0]
        search_result = self.database.search(query_embed_np)
        if len(search_result) >= 0:
            idx = search_result[0].id
            user_name = search_result[0].payload["user_name"]
            score = search_result[0].score
            if score >= self.thresh:
                return {"idx": idx, "user_name": user_name, "score": score}
        return None


if __name__ == "__main__":
    image_path_1 = "/home/son.nguyen/project/face_human_detect/CVLface/test/s3.png"
    image_path_2 = "/home/son.nguyen/project/face_human_detect/CVLface/test/s2.png"
    image_path_3 = "/home/son.nguyen/project/face_human_detect/CVLface/test/x1.png"

    face_recognition = FaceRecognition()
    search_result = face_recognition.inference(Image.open(image_path_1))
    print("ok")
