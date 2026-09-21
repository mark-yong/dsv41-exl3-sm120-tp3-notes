# Jake Tempo source tree rebuilt on amd64 / SM120 (RTX PRO 6000).
# Upstream Tempo Dockerfile pins an ARM64 base + aarch64 cmake wheel + ARCH 12.1a.
# This port keeps the same archives/overlays; natives rebuild for 12.0a.
FROM vllm/vllm-openai:deepseekv41-flash-0909

COPY inputs/ /input/
COPY recipe/build/stage.py recipe/build/run_stage.sh /tempo/build/
RUN mkdir -p /receipts \
 && python3 -m pip install --no-index --no-deps \
      /input/cmake-3.31.6-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

RUN bash /tempo/build/run_stage.sh 1
RUN bash /tempo/build/run_stage.sh 2
RUN bash /tempo/build/run_stage.sh 3
RUN bash /tempo/build/run_stage.sh 4
RUN bash /tempo/build/run_stage.sh 5

COPY recipe/ /tempo/
ARG SOURCES_SHA256
ARG TEMPO_AMD64_PORT=1
LABEL jspark3.sources=$SOURCES_SHA256 \
      tempo_amd64_port=$TEMPO_AMD64_PORT \
      homelab.arch_list=12.0a

# apply_patches.py resolves ROOT relative to tools/; pins live under /tempo
WORKDIR /tempo
RUN python3 /tempo/tools/apply_patches.py

ENV TORCH_CUDA_ARCH_LIST=12.0a \
    FLASHINFER_CUDA_ARCH_LIST=12.0a \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    VLLM_HAS_FLASHINFER_CUBIN=1

ENTRYPOINT ["vllm", "serve"]
CMD []
