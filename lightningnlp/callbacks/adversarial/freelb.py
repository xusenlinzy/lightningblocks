import torch


class FreeLB(object):
    """
    PGD是迭代K次后取最后一次扰动的梯度更新参数，FreeLB是取K次迭代中的平均梯度
    """
    def __init__(self, model, adv_K=3, adv_lr=1e-2, adv_init_mag=2e-2, adv_max_norm=0.,
                 adv_norm_type='l2', base_model='bert'):
        self.model = model
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model

    def attack(self, inputs, gradient_accumulation_steps=1):
        input_ids = inputs['input_ids']
        if isinstance(self.model, torch.nn.DataParallel):
            embeds_init = getattr(self.model.module, self.base_model).embeddings.word_embeddings(input_ids)
        else:
            embeds_init = getattr(self.model, self.base_model).embeddings.word_embeddings(input_ids)
        if self.adv_init_mag > 0:
            input_mask = inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.adv_init_mag, self.adv_init_mag)
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)

        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs['inputs_embeds'] = delta + embeds_init
            inputs['input_ids'] = None
            outputs = self.model(**inputs)
            loss, logits = outputs["losses"], outputs["logits"]  # model outputs are always dict in transformers
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()
            delta_grad = delta.grad.clone().detach()
            if self.adv_norm_type == "l2":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta = torch.clamp(delta, -self.adv_max_norm, self.adv_max_norm).detach()
            else:
                raise ValueError("Norm type {} not specified.".format(self.adv_norm_type))
            if isinstance(self.model, torch.nn.DataParallel):
                embeds_init = getattr(self.model.module, self.base_model).embeddings.word_embeddings(input_ids)
            else:
                embeds_init = getattr(self.model, self.base_model).embeddings.word_embeddings(input_ids)
        return loss
