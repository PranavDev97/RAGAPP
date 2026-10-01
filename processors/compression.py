from llmlingua import PromptCompressor


# constants
COMPRESSION_MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"



llm_lingua : PromptCompressor | None = None
async def create_compression_model():
    global llm_lingua
    llm_lingua = PromptCompressor(
        model_name=COMPRESSION_MODEL_NAME,
        device_map="cpu",
        use_llmlingua2=True, # Whether to use llmlingua-2
    )


async def compress_answer(answer: str) -> str:
    compressed_answer = llm_lingua.compress_prompt(answer, rate=0.6 , force_tokens = ['\n', '?', 'not'])
    return compressed_answer["compressed_prompt"]








