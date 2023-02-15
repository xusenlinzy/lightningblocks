from .model import get_auto_tc_model

from ..predictor import TextClassificationPredictor


def get_auto_tc_predictor(
    model_name="fc",
    model_type="bert",
    model=None,
    model_name_or_path=None,
    tokenizer=None,
    device="cpu",
    use_fp16=False,
    load_weights=True,
) -> TextClassificationPredictor:

    if model is None:
        model = get_auto_tc_model(model_name=model_name, model_type=model_type)

    if model_name not in ["mrc", "lear"]:
        return TextClassificationPredictor(model, model_name_or_path=model_name_or_path, tokenizer=tokenizer,
                                           device=device, load_weights=load_weights, use_fp16=use_fp16)
