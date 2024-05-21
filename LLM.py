from torch import cuda, bfloat16
import transformers
from accelerate import disk_offload

def LLM():
    model_id = 'GreenBitAI/LLaMA-7B-2bit' 

    # Device Selection (GPU if available, otherwise CPU)
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # 1. BitsAndBytes Configuration (Important for 2-bit quantization) only for GPU
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=False, 
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # 2. Initialize Model Config (Optional, but recommended)
    hf_auth = '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'  
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        token=hf_auth,
        pad_token_id=0 
    )
    print("Starting model loading...")
    
    # 3. Load the Model (using BitsAndBytes and disk_offload)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,#quantization_config=bnb_config,  # Apply the BitsAndBytes config
        device_map='auto',
        token=hf_auth,
        offload_folder='offload',  
        force_download=False 
    )
    print("Model loaded successfully.")

    model = disk_offload(model)
    model.eval()
    print(f"Model loaded on {device}")
    return model