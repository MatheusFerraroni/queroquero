# Quero-Quero

Preparação reprodutível de corpora PT-BR para futuro continual pretraining do
`Polygl0t/Tucano2-0.6B-Base`.

## Estado

A etapa de preparação de dados está implementada para os seis datasets. Ela
valida e seleciona documentos, faz limpeza conservadora, deduplicação exata
intradataset, tokenização com o tokenizer original, separação por documento,
packing e escrita de shards verificáveis.

Este repositório ainda não implementa treinamento, uso de GPUs, avaliação de
modelo ou exportação de pesos.

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
| `gigaverbo` | [GigaVerbo-v2](docs/datasets/gigaverbo.md) | `default/train` remoto e pinado |

Os limites e regras de cada fonte ficam em `configs/datasets/`; as regras
comuns ficam em `configs/preparation.json`.

## Instalação

Use Python 3.12 em um ambiente virtual gerenciado pelo `uv`:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

A preparação não depende de PyTorch. Para as cinco fontes locais, copie o
arquivo de ambiente e ajuste a raiz somente leitura uma vez:

```sh
cp .env.example .env
```

`PTBR_OUTPUT_ROOT` é opcional: quando ausente ou definido como `derived`, a
saída fica no repositório. Um caminho relativo é resolvido a partir do projeto;
um caminho absoluto permite armazenar os derivados em outro volume. O `.env` é
local e ignorado pelo Git.

Todos os comandos Python deste projeto passam explicitamente por `uv` e pelo
interpretador `.venv/bin/python`; não use o Python do sistema.

GigaVerbo é lido por streaming e requer acesso à Hugging Face.

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

Para executar o perfil `smoke` dos seis datasets em sequência:

```sh
./scripts/prepare_smoke_all.sh
```

O script localiza a raiz do projeto automaticamente, carrega `.env` pelo `uv`,
usa somente `.venv/bin/python` e continua para os demais datasets quando uma
fonte falha. Ao final, retorna código diferente de zero se houver qualquer
falha.

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

## Revisão de boilerplate do WackyWacky

No perfil `mvp`, `filters.boilerplate.decision` começa como `pending`. A passagem
produz `boilerplate_report.json`, encerra com status de revisão e não publica
shards finais. Após revisar apenas as contagens do relatório, configure uma
decisão explícita:

- `keep`: mantém os parágrafos medidos;
- `remove_exact`: remove parágrafos normalizados com pelo menos 80 caracteres,
  repetidos em pelo menos 5 documentos e 3 domínios.

O relatório não contém exemplos ou conteúdo da fonte.

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

O contrato futuro do modelo permanece documentado em
[`docs/model-artifact-contract.md`](docs/model-artifact-contract.md), mas não é
executado pela entrega atual.
