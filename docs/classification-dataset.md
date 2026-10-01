# Dataset canônico de classificação do Adrenaline

Esta etapa prepara um dataset privado para comparar B0, Geral e Fórum. Ela não
calcula embeddings nem treina classificadores.

## Contrato

As fontes são `forum.adrenaline.com.br.zip`, `conversations.zip` e o manifesto
Adrenaline do CPT pareado, todos fixados por fingerprints em
`configs/classification/adrenaline-v1.json` e abertos somente para leitura.

Qualquer thread ligada a um fragmento presente nos shards CPT de treino ou
avaliação é excluída. O cruzamento precisa resolver todos os hashes; caso
contrário, a preparação falha. Títulos e primeiros posts vazios também são
excluídos. Conteúdo exato repetido com o mesmo rótulo conserva um exemplo;
conflitos de rótulo removem o grupo inteiro.

A limpeza decodifica entidades HTML até estabilizar, com no máximo oito
iterações, e só então remove marcação, caracteres de controle e espaços
redundantes sob NFC. Registros que ultrapassem esse limite são descartados e
contabilizados apenas de forma agregada. Essa política é versionada como
`html_entities_until_stable_max_8` e também se aplica aos nomes dos rótulos.

O resultado não possui coluna ou diretório de split:

```text
$PTBR_CLASSIFICATION_ROOT/adrenaline/<classification-dataset-id>/
├── examples.parquet
├── examples.csv.gz
├── categories.csv
├── subcategories.csv
├── audit.json
└── dataset_manifest.json
```

Parquet e CSV contêm texto privado e permanecem fora do Git. Logs e manifests
registram somente contagens, políticas, IDs derivados e hashes agregados.

## Preparar e validar no Slurm

Configure `PTBR_CLASSIFICATION_ROOT` como caminho absoluto no `.env`. O job é
CPU-only, retomável por blocos e tem limite de 24 horas:

```bash
BUILD_JOB_ID=$(./scripts/submit_classification.sh build)
echo "$BUILD_JOB_ID"
```

Depois de obter o `classification_dataset_id`, valide o diretório completo:

```bash
DATASET_PATH="$PTBR_CLASSIFICATION_ROOT/adrenaline/<classification-dataset-id>"
VALIDATE_JOB_ID=$(
  ./scripts/submit_classification.sh validate-dataset "$DATASET_PATH"
)
echo "$VALIDATE_JOB_ID"
```

Aceite apenas `COMPLETED 0:0` e JSON com `"status": "valid"`.

## Materializar um split experimental

O código do experimento escolhe um exemplo por título equivalente, balanceia
as classes e cria splits estratificados 70/15/15. As seeds permitidas são
42–46. Cada manifesto é independente do modelo e da variante de entrada:

```bash
SPLIT_PATH="$PTBR_CLASSIFICATION_ROOT/splits/coarse/seed-42/split_manifest.json"
SPLIT_JOB_ID=$(
  ./scripts/submit_classification.sh split \
    "$DATASET_PATH" coarse 42 "$SPLIT_PATH"
)
echo "$SPLIT_JOB_ID"
```

Valide o split de forma independente:

```bash
./scripts/submit_classification.sh validate-split \
  "$DATASET_PATH" "$SPLIT_PATH"
```

A tarefa `coarse` usa as seis categorias de conteúdo e até 2.000 exemplos por
classe. A tarefa `fine` usa subcategorias dessas seis categorias que possuam ao
menos 1.000 títulos elegíveis, com exatamente 1.000 exemplos por classe.
