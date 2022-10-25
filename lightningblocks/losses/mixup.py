import random
import torch
import torch.nn as nn
import numpy as np
import warnings


class MixUp(nn.Module):
    """mixup方法实现
        method: embed, encoder分别表示在embedding和encoder层面做mixup, None表示mix后续处理, hidden表示对隐含层做mixup
    """

    def __init__(self, method='encoder', alpha=1.0, layer_mix=None):
        super().__init__()
        assert method in {'embed', 'encoder', 'hidden', None}
        self.method = method
        self.alpha = alpha
        self.perm_index = None
        self.lamb = 0
        self.layer_mix = layer_mix  # 需要mix的隐含层index

    def get_perm(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs[self.perm_index]
        elif isinstance(inputs, (list, tuple)):
            return [value[self.perm_index] if isinstance(value, torch.Tensor) else value for value in inputs]

    def mix_up(self, output, output1):
        if isinstance(output, torch.Tensor):
            return self.lamb * output + (1.0 - self.lamb) * output1
        elif isinstance(output, (list, tuple)):
            output_final = []
            for i in range(len(output)):
                if output[i] is None:  # conditional_emb=None
                    output_final.append(output[i])
                elif (not output[i].requires_grad) and (output[i].dtype in {torch.long, torch.int}):
                    # 不是embedding形式的
                    output_final.append(torch.max(output[i], output1[i]))
                else:
                    output_final.append(self.lamb * output[i] + (1.0 - self.lamb) * output1[i])
            return output_final
        else:
            raise ValueError('Illegal model output')

    def encode(self, model, inputs):
        input_ids = inputs['input_ids']
        batch_size, device = input_ids.shape[0], input_ids.device

        self.lamb = np.random.beta(self.alpha, self.alpha)
        self.perm_index = torch.randperm(batch_size).to(device)

        if self.method is None:
            output = model(**inputs)
            output_perm = self.get_perm(output)
            return [output, output_perm]

        elif self.method == 'encoder':
            output = model(**inputs)
            output_perm = self.get_perm(output)
            output_final = self.mix_up(output, output_perm)

        elif self.method == 'embed':
            output = model.embeddings.word_embeddings(input_ids)
            output_perm = self.get_perm(output)
            output_final = self.mix_up(output, output_perm)
            output_final = model.encoder(output_final)
            # Final
            output_final = model.apply_final_layers(output_final)

        elif self.method == 'hidden':
            if self.layer_mix is None:
                # 这里暂时只考虑encoderLayer, 不考虑decoderLayer和seq2seq模型结构
                try:
                    layer_mix = random.randint(0, len(model.encoderLayer))
                except Exception:
                    warnings.warn('LayerMix random failded')
                    layer_mix = 0
            else:
                layer_mix = self.layer_mix

            def apply_on_layer_end(l_i, output):
                if l_i == layer_mix:
                    output1 = self.get_perm(output)
                    return self.mix_up(output, output1)
                else:
                    return output

            model.apply_on_layer_end = apply_on_layer_end
            output_final = model(inputs)
        return output_final

    def forward(self, y_pred, y_true, loss_fct=nn.CrossEntropyLoss()):
        """ 计算损失
        """
        y_true_perm = y_true[self.perm_index]
        return self.lamb * loss_fct(y_pred, y_true) + (1 - self.lamb) * loss_fct(y_pred, y_true_perm)
