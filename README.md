# Quero-Quero

Preparação e continual pretraining reprodutíveis de corpora PT-BR para o
`Polygl0t/Tucano2-0.6B-Base`.

## Estado

A etapa de preparação de dados está implementada para os seis datasets. Ela
valida e seleciona documentos, faz limpeza conservadora, deduplicação exata
intradataset, tokenização com o tokenizer original, separação por documento,
packing e escrita de shards verificáveis.

O continual pretraining full-parameter, a avaliação por loss/perplexidade,
checkpoints retomáveis e a exportação Hugging Face também estão implementados.
Os alvos executáveis são uma P100 de 12 GB ou duas L40S no mesmo nó. A execução
real continua dependente dos derivados locais e da alocação Slurm do cluster.

## Contrato da preparação

| Item | Valor |
| --- | --- |
| modelo/tokenizer | `Polygl0t/Tucano2-0.6B-Base` |
| revisão | `dad97dc864a8f9a1d240fb9351d098f3af9511d7` |
| tamanho da sequência | 1.024 tokens, sem padding |
| seed | 42 |
| formato | Parquet com compressão ZSTD |
| shard | até 1.024 sequências |
| `smoke` | 8 sequências de treino + 2 de avaliação |
| `mvp` | 256 sequências de treino + 32 de avaliação |

Cada sequência contém somente `sequence_id`, `input_ids`, hashes das
referências de origem e contagens de tokens por referência. Texto, URLs,
identificadores pessoais, `labels` e `attention_mask` não são persistidos nos
shards.

O tokenizer, vocabulário, IDs e special tokens permanecem inalterados. O EOS é
inserido entre documentos; somente a cauda final com menos de 1.024 tokens é
descartada e contabilizada.

## Datasets

| ID da CLI | Dataset | Fonte selecionada |
| --- | --- | --- |
| `brwac` | [BrWaC-CLEAN](docs/datasets/brwac.md) | `BrWac.zip` → `data/*.txt` |
| `wackywacky` | [WackyWacky](docs/datasets/wackywacky.md) | `wacky/pages.tsv` filtrado |
| `multiwoz_ptbr` | [MultiWOZ-PTBR](docs/datasets/multiwoz-ptbr.md) | 17 arquivos `dialogues_*.json` |
| `outerspace` | [OuterSpace](docs/datasets/outerspace.md) | `conversations*.zip` |
| `adrenaline` | [Adrenaline](docs/datasets/adrenaline.md) | `conversations*.zip` |
| `gigaverbo` | [GigaVerbo-v2](docs/datasets/gigaverbo.md) | `gigaverbo-v2/default/train-*.parquet` local e pinado |

Os limites e regras de cada fonte ficam em `configs/datasets/`; as regras
comuns ficam em `configs/preparation.json`.

O snapshot rotulado do Adrenaline também pode ser preparado como dataset
canônico, privado e sem split para a avaliação downstream. O contrato e os
comandos estão em
[`docs/classification-dataset.md`](docs/classification-dataset.md).
A avaliação pareada posterior por embeddings e regressão logística está em
[`docs/classification-evaluation.md`](docs/classification-evaluation.md).
As curvas low-shot, NLL condicional inédita e dose por checkpoints estão em
[`docs/classification-diagnostics.md`](docs/classification-diagnostics.md).

## Instalação

Use Python 3.12 em um ambiente virtual gerenciado pelo `uv` para preparação:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

A preparação não depende de PyTorch. Para as seis fontes locais, copie o
arquivo de ambiente e ajuste a raiz somente leitura uma vez:

```sh
cp .env.example .env
```

`PTBR_OUTPUT_ROOT` é opcional: quando ausente ou definido como `derived`, a
saída fica no repositório. Um caminho relativo é resolvido a partir do projeto;
um caminho absoluto permite armazenar os derivados em outro volume. O `.env` é
local e ignorado pelo Git.

Os comandos de preparação passam explicitamente por `uv` e pelo interpretador
`.venv/bin/python`; não use o Python do sistema. No cluster, o runner de treino
ativa o mesmo ambiente pelo Conda antes de chamar `python` diretamente.

As dependências de GPU ficam separadas para que a preparação continue sem
PyTorch. Na P100, instale o ambiente pinado com:

```sh
./scripts/install_training_dependencies.sh
```

Isso instala `torch==2.7.1+cu118` e `bitsandbytes==0.50.0`. CUDA 11.8 é
intencional: a P100 usa a arquitetura Pascal `sm_60`, removida dos builds CUDA
13.

Para duas L40S, instale apenas as dependências comuns, sem tornar
`bitsandbytes` obrigatório:

```sh
./scripts/install_training_dependencies_l40s.sh
```

GigaVerbo é lido por streaming diretamente dos 224 Parquets locais. A cópia
deve preservar o índice `.cache/huggingface/trees/<revisão>.json` criado pelo
download, usado para confirmar a lista, os tamanhos e os hashes sem reler
centenas de gigabytes.

## Executar

Prepare um dataset e um perfil por vez:

```sh
uv run \
  --env-file .env \
  --python .venv/bin/python \
  -m queroquero.prepare run \
  --dataset brwac \
  --profile smoke
```

O progresso é escrito em `stderr` por fase e por checkpoint, sem registrar
texto, URLs ou identificadores da fonte. O resultado final permanece como JSON
em `stdout`.

Troque `brwac` por qualquer ID da tabela e use `mvp` somente quando a fonte e o
orçamento local tiverem sido conferidos. O perfil `smoke` serve apenas para
validar a engenharia; não é uma amostra de pesquisa.

Cada execução resolvida recebe um `preparation-id` derivado da configuração,
do tokenizer e do fingerprint da fonte:

```text
derived/<dataset>/<preparation-id>/
├── train/shard-*.parquet
├── eval/shard-*.parquet
├── dataset_manifest.json
├── preparation_metrics.json
├── progress.json
└── boilerplate_report.json   # somente WackyWacky MVP
```

Checkpoints intermediários permitem retomar a leitura. A retomada só é aceita
quando configuração e fonte continuam compatíveis; shards são publicados por
renomeação atômica depois da validação. Uma execução completa existente é
revalidada e reutilizada.

Para verificar novamente todos os shards e hashes sem reler a fonte:

```sh
uv run \
  --env-file .env \
  --python .venv/bin/python \
  -m queroquero.prepare validate \
  --path derived/brwac/<preparation-id>
```

## Scripts auxiliares

A pasta `scripts/` contém os comandos locais de preparação e inspeção. Ambos
localizam a raiz do projeto automaticamente, carregam `.env` pelo `uv`, usam
somente `.venv/bin/python` e mantêm o cache do `uv` em `cache/uv` por padrão.

| Script | Finalidade |
| --- | --- |
| `scripts/prepare_smoke_all.sh` | Executar o perfil `smoke` dos seis datasets, medir cada duração e continuar quando uma fonte falhar. |
| `scripts/inspect_preparation.sh` | Validar uma preparação e resumir schema, shards, compressão, contagens e métricas sem imprimir conteúdo. |
| `scripts/inspect_smoke_all.sh` | Localizar o smoke mais recente de cada dataset e decodificar a row 0 do split de treino. |
| `scripts/inspect_preparation.py` | Implementação Python do inspetor; normalmente é chamada pelo wrapper `.sh`. |
| `scripts/install_training_dependencies.sh` | Instalar e conferir o stack P100 no Conda atual. |
| `scripts/install_training_dependencies_l40s.sh` | Instalar e conferir o stack L40S sem bitsandbytes obrigatório. |
| `scripts/submit_p100.sh` | Submeter um modo de treino exigindo partição e nó P100 explícitos. |
| `scripts/train_p100.sbatch` | Executar preflight, smoke, MVP, retomada ou validação dentro do Slurm. |
| `scripts/submit_l40s.sh` | Submeter os modos explícitos Geral/Fórum-Tec em duas L40S, sem fixar nó. |
| `scripts/submit_paired_preparation.sh` | Submeter auditoria, preparação serial e verificação pareada via Slurm. |
| `scripts/train_l40s.sbatch` | Executar os modos distribuídos com `torchrun`. |
| `scripts/submit_classification.sh` | Preparar, validar e criar splits determinísticos do benchmark Adrenaline. |
| `scripts/prepare_classification.sbatch` | Executar a preparação privada CPU-only no Slurm por até 24 horas. |
| `scripts/install_classification_dependencies.sh` | Instalar e conferir o scikit-learn fixado para a avaliação downstream. |
| `scripts/submit_classification_evaluation.sh` | Submeter preflight, embeddings, tuning, avaliação final e relatório pareado. |
| `scripts/classification_embeddings.sbatch` | Extrair embeddings em duas L40S com Gloo e chunks retomáveis. |
| `scripts/classification_probe.sbatch` | Executar os arrays CPU de regressão logística e os relatórios. |
| `scripts/submit_classification_diagnostics.sh` | Submeter coorte, low-shot, NLL por estado e relatório diagnóstico. |
| `scripts/classification_diagnostics_cpu.sbatch` | Executar preparação, low-shot e validações diagnósticas em CPU. |
| `scripts/classification_diagnostics_gpu.sbatch` | Executar preflight e NLL em uma L40S, sem DDP/NCCL. |

O fluxo de auditoria, alocação sem reposição e dois treinos pareados de
aproximadamente 12 horas cada está em
[`docs/training-real-l40s.md`](docs/training-real-l40s.md). Os budgets
`paired_real` só são versionados depois que os seis relatórios de capacidade
do cluster forem alocados e passarem por `materialize-paired-real`; até lá, os
modos pareados do Slurm falham fechados.

Execute todos os smoke tests de preparação:

```sh
./scripts/prepare_smoke_all.sh
```

Para inspecionar e decodificar a primeira sequência de treino dos seis smokes:

```bash
./scripts/inspect_smoke_all.sh
```

O comando retorna código diferente de zero ao final caso qualquer dataset
falhe. Execuções interrompidas podem continuar dos checkpoints compatíveis.

Para validar e inspecionar schema, contagens, compressão, proveniência agregada
e métricas em um único relatório seguro:

```sh
./scripts/inspect_preparation.sh derived/brwac/<preparation-id>
```

O inspetor não imprime texto nem hashes de documentos por padrão. Para uma
avaliação linguística local e explícita, é possível decodificar uma única
sequência:

```sh
./scripts/inspect_preparation.sh \
  derived/brwac/<preparation-id> \
  --decode-sample \
  --split train \
  --row 0
```

O conteúdo decodificado é dado real e não deve ser copiado para logs, testes,
commits ou documentação.

## Revisão de boilerplate do WackyWacky

O perfil `smoke` usa `remove_exact` automaticamente e grava
`boilerplate_report.json` sem interromper a execução. No perfil `mvp`,
as contagens agregadas foram revisadas e a decisão versionada em
`filters.boilerplate.decision_by_profile.mvp` é `remove_exact`. O valor
`pending` permanece como gate de revisão: produz o mesmo relatório agregado,
encerra com status de revisão e não publica shards finais. As decisões finais
possíveis são:

- `keep`: mantém o texto dos candidatos;
- `remove_exact`: remove parágrafos exatos com pelo menos 80 caracteres,
  repetidos em 5 documentos e 3 domínios, e janelas exatas de 3 linhas com ao
  menos 60 caracteres, repetidas em 5 documentos do mesmo domínio.

Depois da normalização, o WackyWacky remove globalmente cada linha não vazia
com menos de 40 caracteres, inclusive quando um documento contém várias
quebras de linha. Páginas identificadas como busca, categoria, tag ou arquivo
são descartadas antes da descompactação usando somente título e URL em memória.
Falhas limitadas ao campo `text` — MD5 ou hexadecimal inválido, payload vazio,
frame Zstandard inválido ou corrompido, limite descomprimido excedido e saída
não UTF-8 — descartam apenas o registro e são contabilizadas por motivo. O TSV
continua estrito, bytes inválidos nunca são substituídos e nenhum campo do
registro é escrito nos logs.

Depois dos filtros, documentos afetados são descartados quando restam menos de
300 caracteres ou quando mais de 80% do texto normalizado foi removido. A troca
de `pending` para uma decisão final reutiliza os candidatos do scan completo;
alterar qualquer limiar inicia um novo scan.

O relatório v2 e as métricas desses filtros não contêm exemplos, URLs, títulos,
hashes de blocos ou conteúdo da fonte.

## Continual pretraining na P100 ou em 2× L40S

O treino usa os seis datasets em proporções iguais, sem LoRA, adapters,
quantização dos pesos ou alteração do tokenizer. Os parâmetros permanecem em
FP32; forward e backward usam autocast FP16 com escala dinâmica. Gradient
checkpointing, microbatch 1 e AdamW8bit mantêm o uso de memória compatível com
a P100 de 12 GB.

Nas duas L40S, o mesmo batch global de oito sequências é dividido em quatro
sequências por rank. O treino usa DDP/NCCL, BF16 sem scaler, AdamW fundido e
três microbatches sob `no_sync()`; o quarto sincroniza o gradiente global. Os
pesos, gradientes e estados do AdamW permanecem FP32.

| Perfil | treino/avaliação por dataset | passos | checkpoint |
| --- | ---: | ---: | ---: |
| `smoke` | 8 / 2 | 6 | 3 |
| `mvp` | 256 / 32 | 192 | 96 |

As configurações `queroquero-training-config/v2` ficam em `configs/training/`.
Antes de qualquer treino, o preflight exige o hardware exato do perfil, wheel
CUDA 11.8, todos os seis manifests atuais e um passo global real sem OOM ou
valores não finitos. O perfil L40S exige dois ranks, duas GPUs homogêneas com
compute capability `sm_89`, kernels CUDA binariamente compatíveis, BF16, NCCL e
AdamW fundido.

O fluxo completo de instalação, cache, submissão, retomada, monitoramento e
validação está em [`docs/training-p100.md`](docs/training-p100.md) e
[`docs/training-l40s.md`](docs/training-l40s.md).

As interfaces principais são:

```sh
python -m queroquero.train cache-model
python -m queroquero.train preflight --config configs/training/p100-smoke.json
python -m queroquero.train run --config configs/training/p100-smoke.json
python -m queroquero.train run --config configs/training/p100-smoke.json --resume
python -m queroquero.train run --config configs/training/p100-mvp.json
python -m queroquero.train validate --artifact artifacts/<artifact-id>
```

O alvo L40S executa `preflight` e `run` exclusivamente por:

```sh
torchrun --standalone --nproc_per_node=2 \
  -m queroquero.train run --config configs/training/l40s-mvp.json
```

Runs, métricas, checkpoints, logs e pesos permanecem fora do Git. O MVP mede
loss e perplexidade por dataset antes e depois do epoch. Uma regressão bloqueia
promoção, mas não apaga o artefato técnico; OOM, NaN, hashes divergentes ou
tokenizer alterado encerram a execução sem fallback silencioso.

## Segurança e proveniência

Fontes são abertas somente para leitura e ZIPs não são extraídos. A separação
treino/avaliação ocorre por hash estável da referência antes do packing, de
modo que um documento ou conversa não atravesse os splits. Deduplicação cruzada
e mistura de datasets não fazem parte desta etapa.

Dados, caches, derivados, manifests de execução e shards permanecem ignorados
pelo Git. Não registre conteúdo de fonte em documentação, fixtures, testes ou
logs. GigaVerbo e todos os derivados continuam
`internal_research_only`; uma eventual exportação dependerá da revisão das
permissões dos corpora.

## Testes

As fixtures sintéticas exercitam os seis adapters e o núcleo comum sem copiar
registros reais:

```sh
.venv/bin/python -m unittest discover -v
```

O exportador e o validador executam o contrato documentado em
[`docs/model-artifact-contract.md`](docs/model-artifact-contract.md).
