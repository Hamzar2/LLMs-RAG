from torch import cuda, bfloat16
import transformers
from accelerate import disk_offload


def LLM():
    model_id = 'GreenBitAI/LLaMA-7B-2bit'

    device = 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # begin initializing HF items, need auth token for these
    hf_auth = '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'
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        token=hf_auth,
        pad_token_id = 0
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config, #quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth,
        offload_folder=r'rag-tutorial-v2\offload',
        force_download=False
    )
    # model = disk_offload(model)
    # model.eval()
    print(f"Model loaded on {device}")

    return model