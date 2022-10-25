import torch
from ...losses import KL


class ALUM(object):
    """
    Adversarial Training for Large Neural Language Models
    """

    def __init__(self, model, adv_lr, adv_K=3, adv_var=1e-5, adv_alpha=1.0, adv_gamma=1e-6,
                 adv_norm_type='inf', base_model='bert'):
        self.model = model
        self.adv_var = adv_var
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_gamma = adv_gamma
        self.adv_alpha = adv_alpha
        self.adv_norm_type = adv_norm_type
        self.kl = KL()
        self.base_model = base_model

    def adv_project(self, grad, eps=1e-06):
        if self.adv_norm_type == 'l2':
            return grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif self.adv_norm_type == 'l1':
            return grad.sign()
        else:
            return grad / (grad.abs().max(-1, keepdim=True)[0] + eps)

    def attack(self, inputs, gradient_accumulation_steps=1):
        input_ids = inputs['input_ids']
        outputs = self.model(**inputs)
        loss, logits = outputs["losses"], outputs["logits"]
        if isinstance(self.model, torch.nn.DataParallel):
            embeds_init = getattr(self.model.module, self.base_model).embeddings.word_embeddings(input_ids)
        else:
            embeds_init = getattr(self.model, self.base_model).embeddings.word_embeddings(input_ids)
        input_mask = inputs['attention_mask'].to(embeds_init)
        delta = torch.zeros_like(embeds_init).normal_(0, 1) * self.adv_var * input_mask.unsqueeze(2)

        for _ in range(self.adv_K):
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init
            inputs['input_ids'] = None
            adv_outputs = self.model(**inputs)
            adv_logits = adv_outputs["logits"]  # model outputs are always tuple in transformers (see doc)

            adv_loss = self.kl(adv_logits, logits.detach())
            delta_grad, = torch.autograd.grad(adv_loss, delta, only_inputs=True)
            adv_direct = self.adv_project(delta_grad, eps=self.adv_gamma)

            inputs['inputs_embeds'] = embeds_init + adv_direct * self.adv_lr
            outputs = self.model(**inputs)
            adv_loss_f = self.kl(outputs[1], logits.detach())
            adv_loss_b = self.kl(logits, outputs[1].detach())
            adv_loss = (adv_loss_f + adv_loss_b) * self.adv_alpha
            loss = loss + adv_loss
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if isinstance(self.model, torch.nn.DataParallel):
                embeds_init = getattr(self.model.module, self.base_model).embeddings.word_embeddings(input_ids)
            else:
                embeds_init = getattr(self.model, self.base_model).embeddings.word_embeddings(input_ids)
        return loss
