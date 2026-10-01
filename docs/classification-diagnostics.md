# Diagnósticos do continual pretraining

Este pipeline é separado da avaliação pareada concluída
`5e9cb26cc8b35bc1b33e` e não altera seus arquivos. O contrato executável está
em `configs/classification/diagnostics-v2.json` e fixa a avaliação-fonte, os
embeddings existentes, o dataset, os dez splits, os três modelos finais, os
dois runs e os seis checkpoints intermediários.

São produzidos três diagnósticos:

- curva low-shot em `coarse/title_first_post`, usando os embeddings já
  validados, budgets aninhados e o mesmo teste para os três modelos;
- NLL do primeiro post condicionado ao título em 1.800 threads inéditas;
- curva de NLL nos passos 0, 13k, 26k, 39k e 52k de cada braço.

O low-shot é exploratório porque reutiliza testes já examinados. A NLL e a
curva usam uma coorte que exclui qualquer `title_group_id` presente no teste
de qualquer um dos dez splits anteriores. Grupos vistos somente no treino ou
na validação do classificador podem entrar porque nenhum backbone é ajustado
nessa etapa. Os intervalos da NLL são obtidos por bootstrap pareado e
estratificado por categoria; não são produzidos p-valores.

O contrato v1 permanece versionado para reproduzir a tentativa que excluía
treino, validação e teste. Ele não é usado pelos launchers atuais.

## Dados privados e identidade

Os outputs ficam fora do Git em:

```text
$PTBR_CLASSIFICATION_ROOT/diagnostics/<diagnostic-id>/
├── resolved_diagnostics.json
├── cohort_capacity.json
├── cohort_manifest.json
├── cohort_validation.json
├── low_shot/
├── scores/
├── private/
└── report/
```

Coorte, IDs, previsões e scores por thread são privados. Manifests, logs e
relatórios agregados não contêm texto, IDs individuais ou caminhos absolutos.
A identidade vincula o checkout Git limpo aos hashes da configuração, dataset,
splits, avaliação-fonte, embeddings, artefatos e checkpoints. Retomada com
qualquer divergência é recusada.

Os checkpoints são carregados somente de `step-*/model`. O estado do
otimizador não é desserializado. Antes da inferência são validados run, braço,
passo, manifesto, inventário, hashes dos arquivos do modelo, ausência de
symlinks, arquitetura e tokenizer.

## Ordem de execução

No cluster, atualize o checkout limpo e confirme as dependências da avaliação:

```bash
cd "$HOME/projects/queroquero"
git pull --ff-only
source "$HOME/activate_queroquero.sh"
./scripts/install_classification_dependencies.sh
git status --short --branch
```

Execute **somente um bloco por vez**. Cada job ou array deve terminar
integralmente com `COMPLETED 0:0` antes de submeter o bloco seguinte; os
comandos abaixo não criam dependências Slurm automaticamente.

Primeiro, audite a capacidade com a política v2:

```bash
CAPACITY_JOB_ID=$(./scripts/submit_classification_diagnostics.sh audit-cohort)
```

Exija `COMPLETED 0:0`, `"status": "sufficient"` e pelo menos 300 exemplos
em cada uma das seis categorias. Depois prepare a coorte:

```bash
COHORT_JOB_ID=$(./scripts/submit_classification_diagnostics.sh prepare-cohort)
```

Após a conclusão do job anterior:

```bash
COHORT_VALIDATE_JOB_ID=$(./scripts/submit_classification_diagnostics.sh validate-cohort)
```

Depois da coorte válida, execute o preflight:

```bash
PREFLIGHT_JOB_ID=$(./scripts/submit_classification_diagnostics.sh preflight)
```

Com o preflight aprovado, execute e aguarde o array low-shot:

```bash
LOW_SHOT_JOB_ID=$(./scripts/submit_classification_diagnostics.sh low-shot-unit)
```

Em seguida, execute e aguarde o array de NLL:

```bash
SCORE_JOB_ID=$(./scripts/submit_classification_diagnostics.sh score-unit)
```

Valide todos os caches antes de gerar o relatório:

```bash
SCORES_VALIDATE_JOB_ID=$(./scripts/submit_classification_diagnostics.sh validate-scores)
```

Por fim, gere e valide o relatório em duas etapas separadas:

```bash
REPORT_JOB_ID=$(./scripts/submit_classification_diagnostics.sh report)
```

Após o relatório concluir:

```bash
REPORT_VALIDATE_JOB_ID=$(./scripts/submit_classification_diagnostics.sh validate-report)
```

O low-shot é um array `0-4%4` em CPU. O scorer é um array `0-8%2`, com uma
L40S por unidade e sem DDP/NCCL. Todos os jobs têm limite Slurm de 24 horas,
Hugging Face offline e `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

A aceitação exige 5/5 unidades low-shot, 9/9 caches cobrindo exatamente 1.800
threads, 16.200 scores finitos e relatório final com `"status": "valid"`.
