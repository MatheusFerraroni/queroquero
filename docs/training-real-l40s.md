# Experimento CPT pareado em duas L40S

O experimento real produz dois modelos a partir do mesmo
`Polygl0t/Tucano2-0.6B-Base` fixado. Cada braço consome 416.000 sequências de
1.024 tokens, batch global 8 e 52.000 passos:

- `general`: BrWaC comum, GigaVerbo, WackyWacky, MultiWOZ e BrWaC extra;
- `forum_tech`: os mesmos quatro pools compartilhados, Adrenaline e OuterSpace.

Cada slot de Adrenaline ou OuterSpace é substituído no braço `general` por uma
sequência exclusiva de BrWaC. O limite Slurm é sempre 24 horas; a duração
observada dos braços pode variar.

## 1. Auditar capacidade via Slurm

Atualize o checkout, ative o ambiente e confirme que `.env` aponta para as
fontes. Submeta os seis scans como um array serial, escolhendo um limite inicial
de documentos apropriado:

```bash
cd "$HOME/projects/queroquero"
source "$HOME/activate_queroquero.sh"
git pull --ff-only

CAPACITY_JOB_ID=$(
  ./scripts/submit_paired_preparation.sh capacity-all <documentos>
)
echo "$CAPACITY_JOB_ID"
```

Os relatórios privados ficam em
`$PTBR_OUTPUT_ROOT/.capacity/<dataset>/<id>/capacity_report.json`. Um relatório
`lower_bound` é suficiente somente quando já comprova a cota solicitada pelo
alocador. Os jobs são retomáveis e não registram textos ou referências de
origem.

Se o alocador pedir expansão de apenas uma fonte, retome somente esse scan:

```bash
./scripts/submit_paired_preparation.sh capacity <dataset> <documentos-maior>
```

## 2. Gerar a alocação pareada

Passe um relatório atual de cada dataset:

```bash
python -m queroquero.prepare allocate-paired-real \
  --report <adrenaline-capacity-report.json> \
  --report <brwac-capacity-report.json> \
  --report <gigaverbo-capacity-report.json> \
  --report <multiwoz-capacity-report.json> \
  --report <outerspace-capacity-report.json> \
  --report <wackywacky-capacity-report.json> \
  --output "$PTBR_OUTPUT_ROOT/.capacity/paired-real-allocation.json"
```

O alocador cria primeiro o braço `forum_tech` por divisão igual, redistribuindo
somente capacidade comprovadamente esgotada. Em seguida exige BrWaC suficiente
para `brwac_common + adrenaline + outerspace`. Uma insuficiência interrompe o
fluxo; nenhuma cota é alterada silenciosamente.

Revise primeiro os budgets, ranges, `allocation_sha256` e
`schedule_template_sha256`. Depois materialize deterministicamente os seis
perfis e os dois configs:

```bash
python -m queroquero.prepare materialize-paired-real \
  --allocation "$PTBR_OUTPUT_ROOT/.capacity/paired-real-allocation.json" \
  --output-config-root configs

git diff -- configs
python -m unittest tests.test_config tests.test_training_config \
  tests.test_paired_plan -v
```

O materializador valida tudo antes de escrever e registra a alocação canônica
em `configs/allocations/paired-real-allocation.json`. Revise e versione juntos:

- `profiles.paired_real` dos seis configs de dataset;
- `configs/training/l40s-real-general.json`;
- `configs/training/l40s-real-forum-tech.json`;
- `configs/allocations/paired-real-allocation.json`.

Os modos de treino permanecem bloqueados enquanto esses arquivos não existirem.

## 3. Preparar e verificar os dados

Depois de publicar os budgets exatos:

```bash
PREPARE_JOB_ID=$(./scripts/submit_paired_preparation.sh prepare-all)
echo "$PREPARE_JOB_ID"
```

O array é serial para reduzir contenção nas fontes. Quando as seis tarefas
terminarem, execute a verificação pareada:

```bash
VERIFY_JOB_ID=$(./scripts/submit_paired_preparation.sh verify)
echo "$VERIFY_JOB_ID"
```

Aceite somente `COMPLETED 0:0` e um JSON com `"status": "valid"`, 416.000
sequências por braço, igualdade dos slots compartilhados e substituição 1:1.

## 4. Preflight, treino e retomada

Execute os dois preflights antes de submeter qualquer treino completo:

```bash
GENERAL_PREFLIGHT_JOB_ID=$(
  env NCCL_P2P_DISABLE=1 ./scripts/submit_l40s.sh real-general-preflight
)
FORUM_PREFLIGHT_JOB_ID=$(
  env NCCL_P2P_DISABLE=1 ./scripts/submit_l40s.sh real-forum-tech-preflight
)
```

Depois de ambos concluírem com loss finita:

```bash
GENERAL_JOB_ID=$(
  env NCCL_P2P_DISABLE=1 ./scripts/submit_l40s.sh real-general
)
FORUM_JOB_ID=$(
  env NCCL_P2P_DISABLE=1 ./scripts/submit_l40s.sh real-forum-tech
)
```

Retome somente o mesmo braço e somente após um checkpoint validado:

```bash
env NCCL_P2P_DISABLE=1 ./scripts/submit_l40s.sh real-general-resume
env NCCL_P2P_DISABLE=1 ./scripts/submit_l40s.sh real-forum-tech-resume
```

Os nomes antigos `real`, `real-preflight` e `real-resume` são rejeitados por
serem ambíguos.

## Aceitação

- os dois configs possuem modelo, seed, treinamento, hardware e execução iguais;
- cada braço conclui 52.000 passos e 425.984.000 tokens;
- checkpoints existem nos passos 13.000, 26.000 e 39.000;
- avaliações baseline/final usam os mesmos seis splits e são finitas;
- os dois artefatos FP32 são exportados e validados offline;
- cada job termina em `COMPLETED 0:0`;
- `NCCL_P2P_DISABLE=1` permanece apenas no ambiente de cada job.

Depois das validações independentes, gere o handoff para a futura classificação:

```bash
python -m queroquero.experiment_report \
  --general-run-dir runs/<run-id-general> \
  --forum-tech-run-dir runs/<run-id-forum-tech> \
  --general-artifact artifacts/<artifact-id-general> \
  --forum-tech-artifact artifacts/<artifact-id-forum-tech> \
  --general-elapsed-seconds <ElapsedRaw-general> \
  --forum-tech-elapsed-seconds <ElapsedRaw-forum-tech> \
  --output runs/paired-experiment-report.json
```

Use `sacct -X -j <job-id> --noheader --format=ElapsedRaw` para obter cada valor.
O relatório contém somente IDs, hashes, hiperparâmetros, tempos e métricas.

A classificação downstream de B0, Geral e Fórum/Tec com cinco seeds não faz
parte desta etapa.
