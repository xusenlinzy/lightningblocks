from lightningblocks.task.named_entity_recognition.auto import get_auto_ner_model
from lightningblocks.task.named_entity_recognition.predictor import NerPredictor
from lightningblocks.task.named_entity_recognition.mrc import PromptNerPredictor
from lightningblocks.task.named_entity_recognition.lear import LearNerPredictor
from lightningblocks.task.named_entity_recognition.w2ner import W2NerPredictor


PREDICTOR_MAP = {
    "mrc": PromptNerPredictor,
    "lear": LearNerPredictor,
    "w2ner": W2NerPredictor,
}


def get_auto_ner_predictor(
    model_name="crf",
    model_type="bert",
    model=None,
    model_name_or_path=None,
    tokenizer=None,
    device="cpu",
    use_fp16=False,
    load_weights=True,
    schema2prompt=None,
) -> NerPredictor:

    predictor_class = PREDICTOR_MAP.get(model_name, NerPredictor)

    if model is None:
        model = get_auto_ner_model(model_name=model_name, model_type=model_type)

    if model_name not in ["mrc", "lear"]:
        return predictor_class(model, model_name_or_path=model_name_or_path, tokenizer=tokenizer,
                               device=device, load_weights=load_weights, use_fp16=use_fp16)

    assert schema2prompt is not None, "schema2prompt must be provided."

    return predictor_class(schema2prompt, model=model, model_name_or_path=model_name_or_path,
                           tokenizer=tokenizer, device=device, load_weights=load_weights, use_fp16=use_fp16)
