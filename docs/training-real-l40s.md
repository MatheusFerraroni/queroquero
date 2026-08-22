# Treino real sem reposição em duas L40S

O perfil `real` usa uma única época, 416.000 sequências de 1.024 tokens e
52.000 passos com batch global 8. Ele só pode ser materializado depois da
auditoria das fontes completas: o repositório não presume que os seis corpora
tenham a mesma capacidade.

## 1. Auditar capacidade

Na headnode, com as fontes e o cache do tokenizer disponíveis:

```bash
cd "$HOME/projects/queroquero"
source "$HOME/activate_queroquero.sh"
export PTBR_OUTPUT_ROOT="$HOME/dataset/llm_datasets_derivated"

python -m queroquero.prepare capacity \
  --dataset adrenaline \
  --candidate-documents <limite-de-documentos>
```

Repita para `brwac`, `gigaverbo`, `multiwoz_ptbr`, `outerspace` e
`wackywacky`. O relatório fica em
`$PTBR_OUTPUT_ROOT/.capacity/<dataset>/<id>/capacity_report.json`, não contém
texto ou referência de fonte e permanece fora do Git.

Um relatório `lower_bound` prova apenas o que já foi medido. Se a alocação
informar que a auditoria está incompleta, aumente `--candidate-documents`
somente para os datasets indicados. Um relatório `exact` indica esgotamento da
fonte selecionada.

## 2. Gerar a alocação canônica

Passe explicitamente um relatório atual de cada dataset:

```bash
python -m queroquero.prepare allocate-real \
  --report <adrenaline-capacity-report.json> \
  --report <brwac-capacity-report.json> \
  --report <gigaverbo-capacity-report.json> \
  --report <multiwoz-capacity-report.json> \
  --report <outerspace-capacity-report.json> \
  --report <wackywacky-capacity-report.json> \
  --output "$PTBR_OUTPUT_ROOT/.capacity/real-allocation.json"
```

O alocador começa pela divisão igual, esgota corpora pequenos e redistribui o
déficit deterministicamente. A soma deve ser 416.000, sem oversampling. Se a
capacidade exata agregada for insuficiente, o erro informa sequências e passos
máximos seguros.

Depois dessa saída ser revisada, os seis budgets, o `allocation_sha256` e o
config `configs/training/l40s-real.json` devem ser versionados juntos. Até isso
acontecer, os modos Slurm reais falham antes de submeter o job.

## 3. Preparar e validar os manifests reais

Após publicar os budgets exatos:

```bash
for dataset in adrenaline brwac gigaverbo multiwoz_ptbr outerspace wackywacky; do
  python -m queroquero.prepare run --dataset "$dataset" --profile real
done
```

A preparação real faz packing incremental e reaproveita shards completos e
idênticos após uma retomada. Os manifests devem ter 256 sequências de avaliação
por dataset e a soma dos splits de treino deve ser 416.000.

## 4. Preflight, treino e retomada

`NCCL_P2P_DISABLE=1` continua sendo um workaround somente do ambiente do job:

```bash
REAL_PREFLIGHT_JOB_ID=$(
  env NCCL_P2P_DISABLE=1 ./scripts/submit_l40s.sh real-preflight
)

REAL_JOB_ID=$(
  env NCCL_P2P_DISABLE=1 ./scripts/submit_l40s.sh real
)
```

O job real recebe limite Slurm de 13 horas, checkpoints nos passos 13.000,
26.000 e 39.000, além do checkpoint solicitado por `SIGUSR1`. Retome somente um
run com checkpoint válido:

```bash
REAL_RESUME_JOB_ID=$(
  env NCCL_P2P_DISABLE=1 ./scripts/submit_l40s.sh real-resume
)
```

## Aceitação

- `real-preflight`: duas L40S, BF16, loader lazy e loss finita;
- treino: 52.000 passos, sem referência repetida, OOM ou NaN;
- avaliações baseline/final finitas para os seis datasets;
- checkpoints e run manifest atualizados;
- artefato FP32 `tucano2-model-artifact/v1` exportado e validado offline;
- `COMPLETED 0:0` no treino e na validação independente.

O quality gate é registrado separadamente. Uma regressão bloqueia promoção
científica, mas não remove o artefato técnico.
