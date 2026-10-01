# Treino distribuído em duas L40S

Este runbook executa continual pretraining full-parameter do
`Polygl0t/Tucano2-0.6B-Base` em duas L40S no mesmo nó. O scheduler escolhe o nó;
não use `--nodelist`.

## 1. Ativar o ambiente e instalar as dependências

Na headnode:

```bash
cd "$HOME/projects/queroquero"
source ~/activate_queroquero.sh
./scripts/install_training_dependencies_l40s.sh
```

O instalador usa o Conda atual e valida PyTorch 2.7.1 com CUDA 11.8,
Transformers e suporte a NCCL. `bitsandbytes` não é dependência do alvo L40S.

## 2. Cachear a revisão fixa do modelo

Execute uma vez em um local com internet e home compartilhada:

```bash
export HF_HOME="$PWD/cache/huggingface"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
python -m queroquero.train cache-model
```

Os jobs de compute voltam a operar offline.

## 3. Rodar o preflight distribuído

```bash
JOB_ID=$(./scripts/submit_l40s.sh preflight)
echo "$JOB_ID"

squeue -j "$JOB_ID" -o "%.18i %.9P %.20j %.2t %.10M %.20R"
tail -F "logs/train-l40s-${JOB_ID}.out"
tail -F "logs/train-l40s-${JOB_ID}.err"
sacct -j "$JOB_ID" --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

O preflight deve identificar exatamente duas L40S homogêneas com compute
capability `sm_89`, kernels CUDA binariamente compatíveis, BF16, NCCL e AdamW
fundido, e concluir um passo global completo. Uma wheel pode fornecer cubins
`sm_80`, `sm_86` ou `sm_89` para esse hardware; o passo real confirma que os
kernels necessários executam. Só prossiga com `COMPLETED 0:0`.

## 4. Validar checkpoint e retomada no smoke

```bash
SMOKE_STOP_JOB=$(./scripts/submit_l40s.sh smoke-stop)
echo "$SMOKE_STOP_JOB"
tail -F "logs/train-l40s-${SMOKE_STOP_JOB}.out"
sacct -j "$SMOKE_STOP_JOB" \
  --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

Essa fase termina deliberadamente no checkpoint do passo 3. Depois:

```bash
SMOKE_RESUME_JOB=$(./scripts/submit_l40s.sh smoke-resume)
echo "$SMOKE_RESUME_JOB"
tail -F "logs/train-l40s-${SMOKE_RESUME_JOB}.out"
tail -F "logs/train-l40s-${SMOKE_RESUME_JOB}.err"
sacct -j "$SMOKE_RESUME_JOB" \
  --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

O smoke deve terminar exatamente no passo 6.

## 5. Rodar o MVP

```bash
MVP_JOB=$(./scripts/submit_l40s.sh mvp)
echo "$MVP_JOB"

squeue -j "$MVP_JOB" -o "%.18i %.9P %.20j %.2t %.10M %.20R"
tail -F "logs/train-l40s-${MVP_JOB}.out"
tail -F "logs/train-l40s-${MVP_JOB}.err"
sacct -j "$MVP_JOB" \
  --format=JobID,JobName,Partition,State,Elapsed,ExitCode,MaxRSS
```

O MVP conclui 192 passos e grava o checkpoint programado no passo 96. Se o
Slurm enviar `SIGUSR1`, o shell cria um marcador, os dois ranks salvam o
checkpoint no próximo limite de passo e o wrapper encerra com código 99. Para
continuar:

```bash
MVP_RESUME_JOB=$(./scripts/submit_l40s.sh mvp-resume)
```

## 6. Validar o artefato offline

Use `artifact.path` do JSON final do MVP:

```bash
export MODEL_ARTIFACT_PATH="$HOME/projects/queroquero/artifacts/<artifact-id>"
VALIDATE_JOB=$(./scripts/submit_l40s.sh validate)
echo "$VALIDATE_JOB"
tail -F "logs/train-l40s-${VALIDATE_JOB}.out"
sacct -j "$VALIDATE_JOB" \
  --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

O validador roda em processo único, relê hashes, rejeita arquivos extras e
symlinks, confirma o tokenizer original e carrega os pesos FP32 offline.

## Critérios finais

- Smoke: checkpoint no passo 3, retomada e término no passo 6.
- MVP: 192 passos, checkpoint no 96 e `COMPLETED 0:0`.
- Métricas baseline/final finitas para os seis datasets, sem texto de fonte.
- Artefato `tucano2-model-artifact/v1` válido e carregável offline.
- Sem fallback para LoRA, contexto menor, outra precisão ou outro modelo.
