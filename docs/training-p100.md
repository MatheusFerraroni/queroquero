# Treino MVP na P100

Este runbook executa continual pretraining full-parameter do
`Polygl0t/Tucano2-0.6B-Base` em uma única P100 de 12 GB. Não use estes comandos
na login node ou na headnode para fazer processamento: somente instalação,
cache e submissão acontecem fora de uma alocação Slurm.

## 1. Conferir a P100 no Slurm

Na headnode:

```bash
sinfo -N -O NodeList:20,Partition:15,Gres:50,StateCompact:12
```

Anote a partição e o nó que realmente expõem a P100. Não reutilize valores de
documentação antiga do cluster.

## 2. Ativar o ambiente e instalar as dependências

```bash
cd "$HOME/projects/queroquero"
source ~/activate_queroquero.sh
./scripts/install_training_dependencies.sh
```

O instalador usa o Python do Conda atual e valida as versões de PyTorch,
CUDA, Transformers e bitsandbytes. Ele não cria outro ambiente.

## 3. Cachear o modelo pinado

Execute uma vez onde o acesso ao Hugging Face seja permitido e o diretório
home seja compartilhado com os compute nodes:

```bash
cd "$HOME/projects/queroquero"
source ~/activate_queroquero.sh

export HF_HOME="$PWD/cache/huggingface"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

python -m queroquero.train cache-model
```

O comando baixa somente a revisão fixa
`dad97dc864a8f9a1d240fb9351d098f3af9511d7`. Os jobs posteriores exigem o
cache e operam offline.

## 4. Configurar a submissão

Na headnode, usando os valores atuais encontrados pelo `sinfo`:

```bash
cd "$HOME/projects/queroquero"
source ~/activate_queroquero.sh

export P100_PARTITION="particao-da-p100"
export P100_NODE="no-da-p100"
```

O wrapper recusa a submissão se um dos dois valores estiver ausente. O
preflight dentro do job também recusa uma GPU cujo nome não contenha `P100` ou
cuja capability não seja `sm_60`.

## 5. Rodar o preflight

```bash
JOB_ID=$(./scripts/submit_p100.sh preflight)
echo "$JOB_ID"

squeue -j "$JOB_ID" -o "%.18i %.9P %.20j %.2t %.10M %.20R"
tail -F "logs/train-${JOB_ID}.out"
tail -F "logs/train-${JOB_ID}.err"
sacct -j "$JOB_ID" --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

Só prossiga quando o JSON final tiver `"status": "ok"` e o `sacct` mostrar
`COMPLETED` com `0:0`.

## 6. Validar checkpoint e retomada no smoke

A primeira fase termina deliberadamente depois do checkpoint do passo 3:

```bash
SMOKE_STOP_JOB=$(./scripts/submit_p100.sh smoke-stop)
echo "$SMOKE_STOP_JOB"
squeue -j "$SMOKE_STOP_JOB"
tail -F "logs/train-${SMOKE_STOP_JOB}.out"
```

Depois de `COMPLETED 0:0`, retome o mesmo run:

```bash
SMOKE_RESUME_JOB=$(./scripts/submit_p100.sh smoke-resume)
echo "$SMOKE_RESUME_JOB"
squeue -j "$SMOKE_RESUME_JOB"
tail -F "logs/train-${SMOKE_RESUME_JOB}.out"
tail -F "logs/train-${SMOKE_RESUME_JOB}.err"
sacct -j "$SMOKE_RESUME_JOB" \
  --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

O smoke termina no passo 6, com avaliações baseline/final. Ele não publica um
artefato de modelo.

## 7. Rodar o MVP

```bash
MVP_JOB=$(./scripts/submit_p100.sh mvp)
echo "$MVP_JOB"

squeue -j "$MVP_JOB" -o "%.18i %.9P %.20j %.2t %.10M %.20R"
tail -F "logs/train-${MVP_JOB}.out"
tail -F "logs/train-${MVP_JOB}.err"
sacct -j "$MVP_JOB" \
  --format=JobID,JobName,Partition,State,Elapsed,ExitCode,MaxRSS
```

Se o Slurm enviar `SIGUSR1`, o processo salva no próximo limite de passo e sai
com código 99. Depois que o job encerrar, retome explicitamente:

```bash
MVP_RESUME_JOB=$(./scripts/submit_p100.sh mvp-resume)
```

Não use `mvp-resume` depois de um run completo.

## 8. Conferir e validar o artefato

O último JSON de `logs/train-${MVP_JOB}.out` informa `artifact.path`,
`artifact_id` e o hash agregado. Use o caminho absoluto correspondente:

```bash
export MODEL_ARTIFACT_PATH="$HOME/projects/queroquero/artifacts/<artifact-id>"

VALIDATE_JOB=$(./scripts/submit_p100.sh validate)
echo "$VALIDATE_JOB"
tail -F "logs/train-${VALIDATE_JOB}.out"
sacct -j "$VALIDATE_JOB" \
  --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

O validador relê todos os hashes, rejeita arquivos extras e symlinks, confirma
o tokenizer e carrega modelo e tokenizer offline.

## Critérios finais

- Smoke: 6 passos, checkpoint no passo 3 e retomada concluída.
- MVP: 192 passos, checkpoint no passo 96 e `COMPLETED 0:0`.
- Baseline e resultado final com loss/perplexidade finitas para os seis
  datasets.
- Artefato `tucano2-model-artifact/v1` válido e carregável offline.
- `quality_gate_passed=false` bloqueia promoção, mesmo com execução técnica
  concluída.
- Não há fallback automático para LoRA, menor contexto, outro modelo ou outra
  precisão.
