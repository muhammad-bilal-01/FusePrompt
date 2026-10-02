import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.evaluation import build_evaluator
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES, PARAPHRASED_TEMPLATES
from clip.model import VisionTransformer, convert_weights
import matplotlib.pyplot as plt
_tokenizer = _Tokenizer()


def diversity_loss(text_features1, text_features2, image_features1, image_features2):

    text_features_similarity = F.cosine_similarity(text_features1, text_features2, dim=1)
    image_features_similarity = F.cosine_similarity(image_features1, image_features2, dim=1)
    text_div_loss = torch.mean(text_features_similarity ** 2)
    image_div_loss = torch.mean(image_features_similarity ** 2)

    div_loss = text_div_loss + image_div_loss    
    return div_loss


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model

def maple_load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def count_trainable_parameters(model):
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_trainable_parameters


def logits_selection(logits_model1, logits_model2):
    logits_model1 = torch.softmax(logits_model1, dim=-1)
    logits_model2 = torch.softmax(logits_model2, dim=-1)

    max_probs_model1, _ = torch.max(logits_model1, dim=-1)
    max_probs_model2, _ = torch.max(logits_model2, dim=-1)

    weights = torch.cat([
        max_probs_model1.unsqueeze(1),
        max_probs_model2.unsqueeze(1)
    ], dim=1)

    weights = torch.softmax(weights, dim=1)

    logits = torch.zeros_like(logits_model1)

    for i in range(logits_model1.shape[0]):
        logits[i] = (
            weights[i][0] * logits_model1[i] +
            weights[i][1] * logits_model2[i]
        )

    return logits

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class MaPLe_TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTSRC.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTSRC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTSRC.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            all_teacher_features_paraphrased = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

            for single_template in PARAPHRASED_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features_paraphrased.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        self.fixed_embeddings_paraphrased = torch.cat(all_teacher_features_paraphrased, dim=1).mean(dim=1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            
            all_teacher_features = []
            all_teacher_features_paraphrased = []
            # Using multiple paraphrased text templates to ensure textual diversity during training            
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

            for single_template in PARAPHRASED_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                # text_features = MaPLe_TextEncoder(clip_model_temp)(x_tokenized.cuda(), None)
                all_teacher_features_paraphrased.append(text_features.unsqueeze(1))
        
        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        self.fixed_embeddings_paraphrased = torch.cat(all_teacher_features_paraphrased, dim=1).mean(dim=1)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        fixed_embeddings = self.prompt_learner.fixed_embeddings
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
            zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
        
            zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()

        return text_features, fixed_embeddings, zero_shot_features, image_features, zero_shot_logits, logit_scale  

class CustomCLIP2(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()        
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        # self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = MaPLe_TextEncoder(clip_model)
        # self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        # prompts = self.prompt_learner()
        # text_features = self.text_encoder(prompts, tokenized_prompts)
        # image_features = self.image_encoder(image.type(self.dtype))
                
        fixed_embeddings_paraphrased = self.prompt_learner.fixed_embeddings_paraphrased
        fixed_embeddings_paraphrased = fixed_embeddings_paraphrased / fixed_embeddings_paraphrased.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
            zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
            
            zero_shot_logits_paraphrased = logit_scale * zero_shot_features.cuda() @ fixed_embeddings_paraphrased.half().cuda().t()

        return text_features, fixed_embeddings_paraphrased, zero_shot_features, image_features, zero_shot_logits_paraphrased, logit_scale

@TRAINER_REGISTRY.register()
class FusePrompt(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTSRC.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        print("Custom: build_model func")
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.evaluator_expert1 = build_evaluator(cfg, lab2cname=self.dm.lab2cname)
        self.evaluator_expert2 = build_evaluator(cfg, lab2cname=self.dm.lab2cname)        

        self.batch_loss = []
        self.batch_loss_ce = []        
        self.batch_loss_nc = []  
        self.batch_loss_scl_text_1 = []
        self.batch_loss_scl_text_2 = []
        self.batch_loss_scl_image_1 = []
        self.batch_loss_scl_image_2 = []
        self.batch_L_SCL_logits_1 = []
        self.batch_L_SCL_logits_2 = []

        self.epoch_losses = []
        self.epoch_losses_ce = []        
        self.epoch_losses_nc = []
        self.epoch_losses_scl_text_1 = []
        self.epoch_losses_scl_text_2 = []
        self.epoch_losses_scl_image_1 = []
        self.epoch_losses_scl_image_2 = []
        self.epoch_L_SCL_logits_1_losses = []
        self.epoch_L_SCL_logits_2_losses = []

        self.epoch_learning_rates = []

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model2 = maple_load_clip_to_cpu(cfg)
        # clip_model2 = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTSRC.PREC == "fp32" or cfg.TRAINER.PROMPTSRC.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            clip_model2.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model2 = CustomCLIP2(cfg, classnames, clip_model2)                
        
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated (1): {enabled}")
        print(f"Parameters count (1): {len(enabled)}")
        print(f"Trainable parameters (1): {count_trainable_parameters(self.model)}")  

        for name, param in self.model2.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)
        
        enabled = set()
        for name, param in self.model2.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated (2): {enabled}")
        print(f"Parameters count (2): {len(enabled)}")
        print(f"Trainable parameters (2): {count_trainable_parameters(self.model2)}")                        
        
        if cfg.MODEL.INIT_WEIGHTS:
            print(f"Loading pretrained weights from {cfg.MODEL.INIT_WEIGHTS}")
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            load_pretrained_weights(self.model2, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model2.to(self.device)

        self.trainable_list = nn.ModuleList([])
        self.trainable_list.append(self.model)
        self.trainable_list.append(self.model2)

        # NOTE: only give prompt_learner to the optimizer
        print("Passing models to dassl")
        self.optim = build_optimizer(self.trainable_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        self.register_model("MaPlePromptLearner", self.model2, self.optim, self.sched)
              
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTSRC.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTSRC.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.model2 = nn.DataParallel(self.model2)
        # Keep model with GPA
        self.previous_model_gpa = None
        self.previous_model_gpa2 = None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)        

        model = self.model
        model2 = self.model2
        optim = self.optim
        scaler = self.scaler
        sched = self.sched

        prec = self.cfg.TRAINER.PROMPTSRC.PREC
        if prec == "amp":
            with autocast():
                loss1 = model(image, label)
                loss2 = model2(image, label)
                loss = loss1 + loss2
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            text_features1, fixed_embeddings, zero_shot_features1, image_features1, zero_shot_logits, logit_scale1 = model(image)
            text_features2, fixed_embeddings_paraphrased, zero_shot_features2, image_features2, zero_shot_logits_paraphrased, logit_scale2 = model2(image)

            text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
            image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
            image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)

            logits1 = logit_scale1 * image_features1 @ text_features1.t()
            logits2 = logit_scale2 * image_features2 @ text_features2.t()            
            
            # weighted_logits = logits_selection(logits1, logits2)
            weighted_logits = (logits1 + logits2) / 2.0
            loss_ce = F.cross_entropy(weighted_logits, label)
            
            div_loss = diversity_loss(text_features1, text_features2, image_features1, image_features2)            
            
            loss_scl_text_1 = F.l1_loss(text_features1, fixed_embeddings.cuda(),
                                      reduction='mean')* 25
                        
            loss_scl_text_2 = F.l1_loss(text_features2, fixed_embeddings_paraphrased.cuda(),
                                      reduction='mean')* 25

            loss_scl_image_1 = F.l1_loss(image_features1, zero_shot_features1.cuda(),
                                      reduction='mean') * 10
            
            loss_scl_image_2 = F.l1_loss(image_features2, zero_shot_features2.cuda(),
                                      reduction='mean') * 10

            L_SCL_logits_1 = F.kl_div(
                F.log_softmax(logits1 / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits1.numel()

            L_SCL_logits_2 = F.kl_div(
                F.log_softmax(logits2 / 1, dim=1),
                F.log_softmax(zero_shot_logits_paraphrased / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits2.numel()

            loss_scl_text = loss_scl_text_1 + loss_scl_text_2
            loss_scl_image = loss_scl_image_1 + loss_scl_image_2
            L_SCL_logits = L_SCL_logits_1 + L_SCL_logits_2

            L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            L_SCL = L_SCL / 2.0  #divide by number of models
            
            loss = (loss_ce + div_loss + L_SCL)
            optim.zero_grad()
            loss.backward()
            optim.step()            

        loss_summary = {"loss": loss.item()}
        # print(f'[{self.batch_idx + 1}/{self.num_batches}] Loss: {loss.item()}')
        self.batch_loss.append(loss.item())
        self.batch_loss_ce.append(loss_ce.item())       
        self.batch_loss_nc.append(div_loss.item()) 
        self.batch_loss_scl_text_1.append(loss_scl_text_1.item())
        self.batch_loss_scl_text_2.append(loss_scl_text_2.item())
        self.batch_loss_scl_image_1.append(loss_scl_image_1.item())
        self.batch_loss_scl_image_2.append(loss_scl_image_2.item())
        self.batch_L_SCL_logits_1.append(L_SCL_logits_1.item())
        self.batch_L_SCL_logits_2.append(L_SCL_logits_2.item())

        if (self.batch_idx + 1) == self.num_batches:
            sched.step()

            first_model = True       
            names = self.get_model_names()        
            for name in names:
              lr = self._optims[name].param_groups[0]['lr']
              print(f"Learning Rate for {name} = {lr}")
              
              if first_model:
                self.epoch_learning_rates.append(lr)
                first_model = False
            
            epoch_loss = sum(self.batch_loss) / self.num_batches
            epoch_loss_ce = sum(self.batch_loss_ce) / self.num_batches 
            epoch_loss_nc =  sum(self.batch_loss_nc) / self.num_batches 
            epoch_loss_scl_text_1 = sum(self.batch_loss_scl_text_1) / self.num_batches
            epoch_loss_scl_text_2 = sum(self.batch_loss_scl_text_2) / self.num_batches
            epoch_loss_scl_image_1 = sum(self.batch_loss_scl_image_1) / self.num_batches
            epoch_loss_scl_image_2 = sum(self.batch_loss_scl_image_2) / self.num_batches
            epoch_L_SCL_logits_1 = sum(self.batch_L_SCL_logits_1) / self.num_batches
            epoch_L_SCL_logits_2 = sum(self.batch_L_SCL_logits_2) / self.num_batches
            
            self.epoch_losses.append(epoch_loss)
            self.epoch_losses_ce.append(epoch_loss_ce)   
            self.epoch_losses_nc.append(epoch_loss_nc)         
            self.epoch_losses_scl_text_1.append(epoch_loss_scl_text_1)
            self.epoch_losses_scl_text_2.append(epoch_loss_scl_text_2)
            self.epoch_losses_scl_image_1.append(epoch_loss_scl_image_1)
            self.epoch_losses_scl_image_2.append(epoch_loss_scl_image_2)
            self.epoch_L_SCL_logits_1_losses.append(epoch_L_SCL_logits_1)
            self.epoch_L_SCL_logits_2_losses.append(epoch_L_SCL_logits_2)
            
            def normalize_loss(loss_list):
              max_loss = max(loss_list)
              return [(loss / max_loss) * 100 for loss in loss_list]

            norm_epoch_losses = normalize_loss(self.epoch_losses)
            norm_epoch_losses_ce = normalize_loss(self.epoch_losses_ce)
            norm_epoch_losses_nc =  normalize_loss(self.epoch_losses_nc)
            norm_epoch_losses_scl_text_1 = normalize_loss(self.epoch_losses_scl_text_1)
            norm_epoch_losses_scl_text_2 = normalize_loss(self.epoch_losses_scl_text_2)
            norm_epoch_losses_scl_image_1 = normalize_loss(self.epoch_losses_scl_image_1)
            norm_epoch_losses_scl_image_2 = normalize_loss(self.epoch_losses_scl_image_2)
            norm_epoch_L_SCL_logits_1_losses = normalize_loss(self.epoch_L_SCL_logits_1_losses)
            norm_epoch_L_SCL_logits_2_losses = normalize_loss(self.epoch_L_SCL_logits_2_losses)
                        
            plt.figure(figsize=(12, 8))
            plt.plot(range(1, len(norm_epoch_losses) + 1), norm_epoch_losses, marker='o', linestyle='-', color='b', label='Total Loss (%)')
            plt.plot(range(1, len(norm_epoch_losses_ce) + 1), norm_epoch_losses_ce, marker='o', linestyle='-', color='r', label='CE Loss (%)')            
            plt.plot(range(1, len(norm_epoch_losses_scl_text_1) + 1), norm_epoch_losses_scl_text_1, marker='o', linestyle='-', color='g', label='SCL Text 1 Loss (%)')
            plt.plot(range(1, len(norm_epoch_losses_scl_text_2) + 1), norm_epoch_losses_scl_text_2, marker='o', linestyle='-', color='c', label='SCL Text 2 Loss (%)')
            plt.plot(range(1, len(norm_epoch_losses_scl_image_1) + 1), norm_epoch_losses_scl_image_1, marker='o', linestyle='-', color='m', label='SCL Image 1 Loss (%)')
            plt.plot(range(1, len(norm_epoch_losses_scl_image_2) + 1), norm_epoch_losses_scl_image_2, marker='o', linestyle='-', color='y', label='SCL Image 2 Loss (%)')
            plt.plot(range(1, len(norm_epoch_L_SCL_logits_1_losses) + 1), norm_epoch_L_SCL_logits_1_losses, marker='o', linestyle='-', color='k', label='SCL Logits 1 Loss (%)')
            plt.plot(range(1, len(norm_epoch_L_SCL_logits_2_losses) + 1), norm_epoch_L_SCL_logits_2_losses, marker='o', linestyle='-', color='brown', label='SCL Logits 2 Loss (%)')
            plt.plot(range(1, len(norm_epoch_losses_nc) + 1), norm_epoch_losses_nc, marker='o', linestyle='--', color='b', label='Div Loss (%)')

            plt.title('Normalized Losses Over Epochs (Percentage)')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (%)')
            plt.legend()
            plt.grid(True)
            save_path = f"./percent_epoch_{len(self.epoch_losses)}.png"
            plt.savefig(save_path)
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(range(1, len(self.epoch_losses) + 1), self.epoch_losses, marker='o', linestyle='-', color='b', label='Total Loss')
            plt.plot(range(1, len(self.epoch_losses_ce) + 1), self.epoch_losses_ce, marker='o', linestyle='-', color='r', label='CE Loss')            
            plt.plot(range(1, len(self.epoch_losses_scl_text_1) + 1), self.epoch_losses_scl_text_1, marker='o', linestyle='-', color='g', label='SCL Text 1 Loss')
            plt.plot(range(1, len(self.epoch_losses_scl_text_2) + 1), self.epoch_losses_scl_text_2, marker='o', linestyle='-', color='c', label='SCL Text 2 Loss')
            plt.plot(range(1, len(self.epoch_losses_scl_image_1) + 1), self.epoch_losses_scl_image_1, marker='o', linestyle='-', color='m', label='SCL Image 1 Loss')
            plt.plot(range(1, len(self.epoch_losses_scl_image_2) + 1), self.epoch_losses_scl_image_2, marker='o', linestyle='-', color='y', label='SCL Image 2 Loss')
            plt.plot(range(1, len(self.epoch_L_SCL_logits_1_losses) + 1), self.epoch_L_SCL_logits_1_losses, marker='o', linestyle='-', color='k', label='SCL Logits 1 Loss')
            plt.plot(range(1, len(self.epoch_L_SCL_logits_2_losses) + 1), self.epoch_L_SCL_logits_2_losses, marker='o', linestyle='-', color='brown', label='SCL Logits 2 Loss')
            plt.plot(range(1, len(self.epoch_losses_nc) + 1), self.epoch_losses_nc, marker='o', linestyle='--', color='b', label='Div Loss')

            plt.title('All Losses Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            save_path = f"./epoch_{len(self.epoch_losses)}.png"
            plt.savefig(save_path)
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(range(1, len(self.epoch_learning_rates) + 1), self.epoch_learning_rates, marker='o', linestyle='-', color='purple', label='Learning Rate')
            plt.title('Learning Rate Over Epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.grid(True)
            save_path = f"./learning_rate_epoch_{len(self.epoch_losses)}.png"
            plt.savefig(save_path)
            plt.close()
            
            self.batch_loss = []
            self.batch_loss_ce = []            
            self.batch_loss_nc = []
            self.batch_loss_scl_text_1 = []
            self.batch_loss_scl_text_2 = []
            self.batch_loss_scl_image_1 = []
            self.batch_loss_scl_image_2 = []
            self.batch_L_SCL_logits_1 = []
            self.batch_L_SCL_logits_2 = []

            # Means one epoch is completed, perform GPA
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            
            # first model
            current_model_weights = copy.deepcopy(model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

            # second model
            current_model_weights2 = copy.deepcopy(model2.state_dict())
            weighted_state_dict2 = self.state_dict_weighting(current_model_weights2, current_epoch_weight)

            if self.previous_model_gpa2 is None:
                self.previous_model_gpa2 = weighted_state_dict2
            else:
                self.previous_model_gpa2 = self.state_dict_add(weighted_state_dict2, self.previous_model_gpa2)

        if self.step_counter == self.model.total_epochs + 1:
            print("Using GPA model for final inference...")
            model.load_state_dict(self.previous_model_gpa)
            self.model.load_state_dict(self.previous_model_gpa)

            model2.load_state_dict(self.previous_model_gpa2)
            self.model2.load_state_dict(self.previous_model_gpa2)

        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        print("Custom: load_model func")
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    
    
    @torch.no_grad()
    def test(self, split=None):
        print("Custom: test func")
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        self.evaluator_expert1.reset()
        self.evaluator_expert2.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = self.parse_batch_test(batch)            

            with torch.no_grad():
                text_features1, fixed_embeddings, zero_shot_features1, image_features1, zero_shot_logits, logit_scale1 = self.model(image)
                text_features2, fixed_embeddings_paraphrased, zero_shot_features2, image_features2, zero_shot_logits_paraphrased, logit_scale2 = self.model2(image)

            text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
            image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
            image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)

            output_expert1 = logit_scale1 * image_features1 @ text_features1.t()
            output_expert2 = logit_scale2 * image_features2 @ text_features2.t()            
            output_combined = (output_expert1 * 0.5 + output_expert2 * 0.5)            
    
            self.evaluator.process(output_combined, label)
            self.evaluator_expert1.process(output_expert1, label)
            self.evaluator_expert2.process(output_expert2, label)

        print("Ensemble:")
        results = self.evaluator.evaluate()
        print("Model1:")
        results_expert1 = self.evaluator_expert1.evaluate()
        print("Model2:")
        results_expert2 = self.evaluator_expert2.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]