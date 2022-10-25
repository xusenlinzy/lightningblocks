from lightningblocks.task.relation_extraction.auto.model import get_auto_re_model
from lightningblocks.task.relation_extraction.predictor import RelationExtractionPredictor


def get_auto_re_predictor(
    model_name="casrel",
    model_type="bert",
    model=None,
    model_name_or_path=None,
    tokenizer=None,
    device="cpu",
    use_fp16=False,
    load_weights=True,
) -> RelationExtractionPredictor:

    if model is None:
        model = get_auto_re_model(model_name=model_name, model_type=model_type)

    return RelationExtractionPredictor(model=model, model_name_or_path=model_name_or_path, tokenizer=tokenizer,
                                       device=device, load_weights=load_weights, use_fp16=use_fp16)
