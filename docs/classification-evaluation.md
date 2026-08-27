# Avaliação pareada por embeddings

Esta etapa compara B0, Geral e Fórum/Tec com regressão logística sobre os dez
splits validados do dataset `dc4b2ce164eab81812a2`. O desfecho principal é
macro-F1 em `coarse` usando título e primeiro post. As demais combinações são
secundárias.

O contrato científico está em
`configs/classification/evaluation-v1.json`. Ele fixa modelos, artefatos,
splits, relatório pareado, comprimento de 1.024 tokens, poolings, grade de `C`,
seeds e contrastes. O preflight exige Git limpo e incorpora o commit à
identidade da avaliação; qualquer divergência impede a retomada.

## Dados privados e outputs

O dataset e os splits permanecem sob `PTBR_CLASSIFICATION_ROOT`. Embeddings,
previsões e relatórios ficam em:

```text
$PTBR_CLASSIFICATION_ROOT/evaluations/<evaluation-id>/
├── resolved_evaluation.json
├── preflight.json
├── embeddings/
├── tuning/
├── selection.json
├── evaluation_units/
├── private/predictions/
└── report/
```

Chunks de embeddings e previsões contêm IDs derivados e nunca devem ser
commitados. Logs e manifests contêm somente contagens, dimensões e digests.

## Ordem de execução no cluster

Atualize o checkout, ative o projeto e instale a dependência adicional:

```bash
cd "$HOME/projects/queroquero"
git pull --ff-only
source "$HOME/activate_queroquero.sh"
./scripts/install_classification_dependencies.sh
git status --short --branch
```

O checkout deve estar limpo. Execute o preflight real em duas L40S:

```bash
PREFLIGHT_JOB_ID=$(
  env NCCL_P2P_DISABLE=1 \
  ./scripts/submit_classification_evaluation.sh preflight
)
echo "$PREFLIGHT_JOB_ID"
```

Aceite somente `COMPLETED 0:0` e JSON com `"status": "ok"`. Em seguida gere
um cache por modelo. Os três jobs podem ser submetidos juntos:

```bash
BASE_JOB_ID=$(env NCCL_P2P_DISABLE=1 ./scripts/submit_classification_evaluation.sh embed-base)
GENERAL_JOB_ID=$(env NCCL_P2P_DISABLE=1 ./scripts/submit_classification_evaluation.sh embed-general)
FORUM_JOB_ID=$(env NCCL_P2P_DISABLE=1 ./scripts/submit_classification_evaluation.sh embed-forum)
echo "base=$BASE_JOB_ID general=$GENERAL_JOB_ID forum=$FORUM_JOB_ID"
```

Depois dos três `COMPLETED 0:0`, valide conjuntamente os caches:

```bash
EMBED_VALIDATE_JOB_ID=$(./scripts/submit_classification_evaluation.sh validate-embeddings)
```

Somente após `"status": "valid"`, execute as 20 unidades de tuning, selecione
um único `(pooling, C)` por tarefa e entrada, e então abra o teste nas 20
unidades finais:

```bash
TUNE_JOB_ID=$(./scripts/submit_classification_evaluation.sh tune-unit)
SELECT_JOB_ID=$(./scripts/submit_classification_evaluation.sh select-hyperparameters)
EVALUATE_JOB_ID=$(./scripts/submit_classification_evaluation.sh evaluate-unit)
```

Cada array precisa terminar integralmente com `COMPLETED 0:0` antes do comando
seguinte. Por fim, gere e valide o relatório:

```bash
REPORT_JOB_ID=$(./scripts/submit_classification_evaluation.sh report)
REPORT_VALIDATE_JOB_ID=$(./scripts/submit_classification_evaluation.sh validate-report)
```

A aceitação final exige 60 avaliações, `"status": "valid"` e ausência de
falhas de convergência. O relatório registra accuracy, macro-F1, métricas por
classe, matrizes de confusão, média, desvio-padrão amostral, IC t de 95% e os
três deltas pareados no sentido “segundo menos primeiro”. Não são calculados
p-valores.

Todos os jobs têm limite Slurm de 24 horas. `NCCL_P2P_DISABLE=1` é passado
somente nos jobs GPU, sem persistência no projeto ou no `.env`.

Durante jobs longos, acompanhe estado, recursos e disco sem abrir arquivos
privados:

```bash
sacct -X -j "$JOB_ID" --format=JobID,JobName,State%14,Elapsed,ExitCode,MaxRSS
du -sh "$PTBR_CLASSIFICATION_ROOT/evaluations"
df -h "$PTBR_CLASSIFICATION_ROOT"
```
