"""
1 - Instalar libmagic (python-magic)

2 - Necessário instalar:
import nltk
nltk.download()

3 - Instalar versão < 2.0.0 do Numpy (1.26.4 recomendado)
4 - Rodar script setando variável PYTHONUTF8=1 (necessário no Windows)
"""

"""
This file contains the code used to process and create the
FineWeb dataset (https://huggingface.co/datasets/HuggingFaceFW/fineweb)
"""
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashConfig, MinhashDedupSignature, MinhashDedupBuckets, \
    MinhashDedupFilter
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import WarcReader, JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter

"""
    we first ran the following pipeline for each dump
"""
DUMP_TO_PROCESS = "CC-MAIN-2023-50"  # example

MAIN_OUTPUT_PATH = "/datatrove/fine"
FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"

main_processing_executor = LocalPipelineExecutor(
    pipeline=[
        WarcReader(
            f"/datatrove/warcs",
            default_metadata={"dump": DUMP_TO_PROCESS},
        ),
        URLFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/1_url/{DUMP_TO_PROCESS}")),
        Trafilatura(favour_precision=True),
        LanguageFilter(
            exclusion_writer=JsonlWriter(
                f"{FILTERING_OUTPUT_PATH}/2_non_english/",
                output_filename="${language}/" + DUMP_TO_PROCESS + "/${rank}.jsonl.gz",
                # folder structure: language/dump/file
            )
        ),
        GopherRepetitionFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/3_gopher_rep/{DUMP_TO_PROCESS}")
        ),
        GopherQualityFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/4_gopher_qual/{DUMP_TO_PROCESS}")
        ),
        C4QualityFilter(
            filter_no_terminal_punct=False,
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/5_c4/{DUMP_TO_PROCESS}"),
        ),
        FineWebQualityFilter(
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/6_fineweb_qual/{DUMP_TO_PROCESS}")
        ),
        JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"),
    ],
    tasks=1,
    logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP_TO_PROCESS}",
)
main_processing_executor.run()

"""
    we then applied minhash deduplication to each individual dump,
"""

# you can also change ngrams or the number of buckets and their size here
minhash_config = MinhashConfig(
    use_64bit_hashes=True,  # better precision -> fewer false positives (collisions)
    num_buckets=1,
    hashes_per_bucket=8,
    n_grams=5,
)

S3_MINHASH_BASE_PATH = f"{MAIN_OUTPUT_PATH}/minhash"

S3_LOGS_FOLDER = f"{MAIN_OUTPUT_PATH}/logs/minhash"
LOCAL_LOGS_FOLDER = "logs/minhash"

TOTAL_TASKS = 1000

# this is the original data that we want to deduplicate
INPUT_READER = JsonlReader(
    f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"
)  # this is the output from the first part

# stage 1 computes minhash signatures for each task (each task gets a set of files)
stage1 = LocalPipelineExecutor(
    pipeline=[
        INPUT_READER,
        MinhashDedupSignature(
            output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/signatures", config=minhash_config
        ),
    ],
    tasks=1,
    logging_dir=f"{S3_LOGS_FOLDER}/signatures",
    depends=main_processing_executor,  # only start after the first one completes
)

stage2 = LocalPipelineExecutor(
    pipeline=[
        MinhashDedupBuckets(
            input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/signatures",
            output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/buckets",
            config=minhash_config,
        ),
    ],
    tasks=1,  # the code supports parallelizing each bucket. here we run 50
    # workers per bucket
    logging_dir=f"{S3_LOGS_FOLDER}/buckets",
    depends=stage1,
)

stage3 = LocalPipelineExecutor(
    pipeline=[
        MinhashDedupCluster(
            input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/buckets",
            output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/remove_ids",
            config=minhash_config,
        ),
    ],
    tasks=1,  # this step runs on a single task
    logging_dir=f"{S3_LOGS_FOLDER}/clustering",
    depends=stage2,
)

stage4 = LocalPipelineExecutor(
    pipeline=[
        INPUT_READER,
        TokensCounter(),  # you can remove this one, it's just a nice way to know how many tokens we have
        # before and after dedup
        MinhashDedupFilter(input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/remove_ids"),
        # run the PII removal
        PIIFormatter(),
        JsonlWriter(f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/deduped_output"),
    ],
    tasks=1,
    logging_dir=f"{S3_LOGS_FOLDER}/filtering",
    depends=stage3,
)

stage4.run()
