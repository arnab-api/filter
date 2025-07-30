import argparse
import json
import logging
import os
from typing import List, Dict
import torch
import transformers
from src.models import ModelandTokenizer
from src.selection.data import SelectionSample, get_random_sample
from src.utils import env_utils, experiment_utils, logging_utils
from src.selection.data import load_people_by_category_fakeverse, load_people_by_category
from src.selection.data import SelectionSample, get_random_sample
from src.functional import predict_next_token, free_gpu_cache, generate_with_patch
from dataclasses import dataclass
from src.utils.typing import PredictedToken
from src.utils.training_utils import TrainableLM_delta
from dataclasses_json import DataClassJsonMixin
from src.rome.rome_hparams import ROMEHyperParams
from src.rome.rome_main import apply_rome_to_model, save_weights, restore_weights


logger = logging.getLogger(__name__)


logger.info(f"{torch.__version__=}, {torch.version.cuda=}")
logger.info(
    f"{torch.cuda.is_available()=}, {torch.cuda.device_count()=}, {torch.cuda.get_device_name()=}"
)
logger.info(f"{transformers.__version__=}")


@dataclass
class SelectionROMEResults(DataClassJsonMixin):
    sample: SelectionSample
    prediction: list[PredictedToken]
    is_correct: bool
    ans_rank: int
    hparams: ROMEHyperParams
    generations: List[str] | Dict


# @torch.inference_mode()
def test_selection_rome_all(
    mt: ModelandTokenizer,
    hparams: ROMEHyperParams,
    limit: int = 12,
    attribute_type: str = "profession",
    save_dir: str | None = None,
    n_distractors: int = 5,
    save_step: int = 5,
):
    """Cache last token states for selection samples."""
    real_people_by_category = load_people_by_category(
        tokenizer=mt.tokenizer,
        category=attribute_type,
    )

    synth_people_by_category = load_people_by_category_fakeverse(
        tokenizer=mt.tokenizer,
        category="occupation",
    )

    os.makedirs(save_dir, exist_ok=True)

    results = []
    n_correct = 0
    prompt_template = "{} is by profession a"
    while len(results) < limit:
        sample = get_random_sample(
            people_by_category=synth_people_by_category,
            mt=mt, 
            n_distractors=n_distractors,
            filter_by_lm_prediction=False,
            #exclude_distractor_categories=["news anchor", "journalist", "entrepreneur", "comedian"],
        )

        # Set the desired information edit
        # Create reverse mapping from entity to category
        entity_to_category = {}
        for category, entities in synth_people_by_category.items():
            for entity in entities.values:  # Use .values to get the actual entity names
                entity_to_category[entity] = category
        
        requests = []
        for entity in [sample.subj] + sample.options:
            # Get the attribute for this specific entity
            entity_attribute = entity_to_category.get(entity)
            requests.append(
                {
                    "prompt": prompt_template,
                    "subject": entity,
                    "target_new": {"str": entity_attribute}
                }
            )

        # Apply the ROME edit
        ## Inject the information edit into the context templates and do a rank one edit.
        for request in requests:
            if request == requests[0]:
                model, orig_weights = apply_rome_to_model(
                    mt=mt,
                    requests=request,
                    hparams=hparams,
                    return_orig_weights=True
                )
            else:
                model = apply_rome_to_model(
                    mt=mt,
                    requests=request,
                    hparams=hparams,
                    return_orig_weights=False
                )

        multi_rome_weights = save_weights(model, list(orig_weights.keys()))

        restore_weights(mt._model, multi_rome_weights)

        # Check
        generations_per_entity = []
        for entity in [sample.subj] + sample.options:
            generation_prompts = [
                f"{entity} is a professional",
                f"What is {entity} known for? {entity} is a",
                f"{entity} is a well-known",
                f"{entity} is a famous",
                f"What is the profession of {entity}? {entity} is a",
            ]

            generations = []
            for prompt in generation_prompts:
                generation = generate_with_patch(
                    mt=mt,
                    inputs=prompt,
                    tokenizer=mt.tokenizer,
                    n_gen_per_prompt=1
                )[0]
                generations.append(generation)

            generations_per_entity.append({
                "entity": entity,
                "generations": generations
            })

        #############
        # MAKING SURE THE REAL ENTITY INFO ISN'T GETTING OVERWRITTEN:
        # for category, entities in real_people_by_category.items():
        #     check_real_entities = []
        #     for entity in entities.values:
        #         generation_prompts = [
        #             f"{entity} is a professional",
        #             f"What is {entity} known for? {entity} is a",
        #         ]                   
        #         
        #         generations = []
        #         for prompt in generation_prompts:
        #             generation = generate_with_patch(
        #                 mt=mt,
        #                 inputs=prompt,
        #                 tokenizer=mt.tokenizer,
        #                 n_gen_per_prompt=1
        #             )[0]
        #             print(generation)
        #             generations.append(generation)                   
        #         
        #         check_real_entities.append({
        #             "entity": entity,
        #             "generations": generations
        #         })
        # print(check_real_entities)
        #############

        prediction, track_ans = predict_next_token(
            mt=mt, inputs=sample.prompt, k=5, token_of_interest=[sample.obj_token_id]
        )
        top_prediction = prediction[0][0]
        if top_prediction.token_id == sample.obj_token_id:
            n_correct += 1

        results.append(
            SelectionROMEResults(
                sample=sample,
                prediction=prediction[0],
                is_correct=top_prediction.token_id == sample.obj_token_id,
                ans_rank=track_ans[0][sample.obj_token_id][0],
                hparams=hparams,
                generations=generations_per_entity,
            )
        )
        if len(results) % save_step == 0 or len(results) == limit:
            logger.info(
                f"Cached {len(results)} samples so far, accuracy={n_correct / len(results) : .3f}  ({n_correct}/{len(results)})."
            )
            with open(os.path.join(save_dir, f"{attribute_type}_results.json"), "w") as f:
                json.dump(
                    dict(
                        accuracy=n_correct / len(results),
                        samples=[r.to_dict() for r in results],
                    ),
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            logger.info(
                f"Saved results to {save_dir}/{attribute_type}_results.json"
            )


#! python -m test_suite.test_01_real_entities --model="meta-llama/Llama-3.3-70B-Instruct" --limit="1000"
#! append "|& tee <log_path>" to save execution logs
if __name__ == "__main__":

    # loading the model
    mt = ModelandTokenizer(
        model_key="meta-llama/Llama-3.3-70B-Instruct",
        #model_key="meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    parser = argparse.ArgumentParser(
        description="Cache selection states for language models"
    )
    logging_utils.add_logging_args(parser)
    experiment_utils.add_experiment_args(parser)

    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Number of samples to generate and cache",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="selection/test_5_rome_all",
        help="Directory to save test results",
    )

    parser.add_argument(
        "--attr",
        type=str,
        default="profession",
        help="Attribute Type",
    )

    parser.add_argument(
        "--n_distractors",
        type=int,
        default=5,
        help="Number of distractors to use",
    )

    parser.add_argument(
        "--save_step",
        type=int,
        default=1,
        help="Save results every N samples",
    )

    parser.add_argument("--layers", type=int, nargs='+', default=[13], 
                        help="List of layer indices to modify")

    parser.add_argument("--fact-token", type=str, default="subject_last",
                        help="Token position for fact extraction")

    parser.add_argument("--v-num-grad-steps", type=int, default=50,
                        help="Number of gradient steps for value optimization")

    parser.add_argument("--v-lr", type=float, default=5e-1,
                        help="Learning rate for value optimization")

    parser.add_argument("--v-loss-layer", type=int, default=None,
                        help="Layer index for computing value loss (default: model.n_layer - 1)")

    parser.add_argument("--v-weight-decay", type=float, default=0.5,
                        help="Weight decay for value optimization")

    parser.add_argument("--clamp-norm-factor", type=float, default=3,
                        help="Factor for clamping gradient norms")

    parser.add_argument("--kl-factor", type=float, default=0.0625,
                        help="KL divergence factor in loss computation")

    parser.add_argument("--mom2-adjustment", action="store_true", default=True,
                        help="Enable second moment adjustment")

    parser.add_argument("--no-mom2-adjustment", dest="mom2_adjustment", action="store_false",
                        help="Disable second moment adjustment")

    parser.add_argument("--context-template-length-params", type=str, 
                        default="[[25, 5], [50, 5]]",
                        help="Context template length parameters as JSON string")

    parser.add_argument("--rewrite-module-tmp", type=str, default=mt.mlp_module_name_format + ".down_proj",
                        help="Template for rewrite module (default: model.mlp_module_name_format + '.down_proj')")

    parser.add_argument("--layer-module-tmp", type=str, default=mt.layer_name_format,
                        help="Template for layer module (default: model.layer_name_format)")

    parser.add_argument("--mlp-module-tmp", type=str, default=mt.mlp_module_name_format,
                        help="Template for MLP module (default: model.mlp_module_name_format)")

    parser.add_argument("--attn-module-tmp", type=str, default=mt.attn_module_name_format,
                        help="Template for attention module (default: model.attn_module_name_format)")

    parser.add_argument("--ln-f-module", type=str, default=mt.final_layer_norm_name,
                        help="Final layer norm module name (default: model.final_layer_norm_name)")

    parser.add_argument("--lm-head-module", type=str, default=mt.lm_head_name,
                        help="LM head module name (default: model.lm_head_name)")

    parser.add_argument("--mom2-dataset", type=str, default="wikipedia",
                        help="Dataset for second moment statistics")

    parser.add_argument("--mom2-n-samples", type=int, default=1000,
                        help="Number of samples for second moment estimation")

    parser.add_argument("--mom2-dtype", type=str, default="float32",
                        help="Data type for second moment computation")

    args = parser.parse_args()
    logging_utils.configure(args)
    experiment_utils.setup_experiment(args)

    logger.info(f"Arguments: {args}")

    # Set v_loss_layer default after mt is available
    if args.v_loss_layer is None:
        args.v_loss_layer = mt.n_layer - 1

    hparams = ROMEHyperParams(
        layers=args.layers,
        fact_token=args.fact_token,
        v_num_grad_steps=args.v_num_grad_steps,
        v_lr=args.v_lr,
        v_loss_layer=args.v_loss_layer,
        v_weight_decay=args.v_weight_decay,
        clamp_norm_factor=args.clamp_norm_factor,
        kl_factor=args.kl_factor,
        mom2_adjustment=args.mom2_adjustment,
        context_template_length_params=json.loads(args.context_template_length_params),
        rewrite_module_tmp=args.rewrite_module_tmp,
        layer_module_tmp=args.layer_module_tmp,
        mlp_module_tmp=args.mlp_module_tmp,
        attn_module_tmp=args.attn_module_tmp,
        ln_f_module=args.ln_f_module,
        lm_head_module=args.lm_head_module,
        mom2_dataset=args.mom2_dataset,
        mom2_n_samples=args.mom2_n_samples,
        mom2_dtype=args.mom2_dtype,
    )

    # Setup cache directory
    save_dir = os.path.join(
        env_utils.DEFAULT_RESULTS_DIR,
        args.save_dir,
        mt.name.split("/")[-1],
    )
    os.makedirs(save_dir, exist_ok=True)

    test_selection_rome_all(
        mt=mt,
        limit=args.limit,
        attribute_type=args.attr,
        save_dir=save_dir,
        n_distractors=args.n_distractors,
        save_step=args.save_step,
        hparams=hparams,
    )